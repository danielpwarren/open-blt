import argparse
import datetime
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch as pt
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from blt.data.jsonl_dataset import JsonlDataset, JsonlValidationDataset
from blt.model.common import get_num_flop_per_token
from blt.model.entropy import EntropyModel
from blt.tokenizer.byte_tokenizer import ByteTokenizer
from blt.utils.checkpoint import save_checkpoint
from blt.utils.logging import init_wandb, log_metrics, log_model_summary, setup_logging
from blt.utils.optim import build_optimizer, get_cosine_schedule

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_text_sample(model, prompt, max_new_tokens, temperature):
    """
    Generate tokens autoregressively. For the first 128 tokens generated,
    force continuation (i.e. ignore EOS) so that a sample is produced.
    """
    model.eval()
    if prompt.size(1) == 0:
        prompt = pt.tensor([[257]], dtype=pt.long).cuda()
    generated = prompt.clone()
    forced_tokens = 128
    with pt.no_grad():
        for i in range(max_new_tokens):
            logits = model(generated)
            next_logits = logits[0, -1]
            scaled_logits = next_logits / temperature
            probabilities = pt.softmax(scaled_logits, dim=-1)
            next_token = pt.multinomial(probabilities, num_samples=1).unsqueeze(0)
            generated = pt.cat([generated, next_token], dim=1)
            if i >= forced_tokens and next_token.item() == 258:
                break
    model.train()
    return generated


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Train BLT Entropy Model on JSONL dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="blt/configs/entropy.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "overrides", nargs="*", help="Optional config overrides in dot list format"
    )
    args = parser.parse_args()

    base_config = OmegaConf.load(args.config)
    config = (
        OmegaConf.merge(base_config, OmegaConf.from_dotlist(args.overrides))
        if args.overrides
        else base_config
    )

    log_dir = os.path.join("logs", os.path.basename(args.config).split(".")[0])
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file=log_file)

    logger.info(f"Full Config: {config}")

    checkpoint_dir = os.path.join(config.dump_dir, config.name)

    init_wandb(config)
    logger.info(f"Run name: {config.name}")

    seed_everything(config.seed)

    tokenizer = ByteTokenizer()

    # Create training dataset
    train_dataset = JsonlDataset(
        data_dir=config.data.train_dir,
        seq_len=config.data.seq_len,
        shuffle_files=True,
        seed=config.seed,
    )

    # Create validation dataset
    val_dataset = JsonlValidationDataset(
        val_file=config.data.val_file,
        seq_len=config.data.seq_len,
        max_samples=config.data.max_val_samples,
    )

    # Calculate total steps if cache doesn't exist or config has changed
    steps_cache_file = os.path.join(config.dump_dir, "steps_cache.json")
    cache_config = {
        "train_dir": config.data.train_dir,
        "seq_len": config.data.seq_len,
        "batch_size": config.data.batch_size,
    }

    if os.path.exists(steps_cache_file):
        with open(steps_cache_file, "r") as f:
            cache = json.load(f)
            if cache["config"] == cache_config:
                total_steps = cache["total_steps"]
                logger.info(f"Using cached total steps: {total_steps}")
            else:
                logger.info("Config changed, recalculating total steps...")
                total_steps = train_dataset.calculate_total_steps(
                    config.data.batch_size
                )
                os.makedirs(config.dump_dir, exist_ok=True)
                with open(steps_cache_file, "w") as f:
                    json.dump({"config": cache_config, "total_steps": total_steps}, f)
    else:
        logger.info("Calculating total steps for the first time...")
        total_steps = train_dataset.calculate_total_steps(config.data.batch_size)
        os.makedirs(config.dump_dir, exist_ok=True)
        with open(steps_cache_file, "w") as f:
            json.dump({"config": cache_config, "total_steps": total_steps}, f)

    logger.info(f"Total training steps: {total_steps}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
    )

    # Create validation iterator
    val_batch_size = 8  # Small fixed batch size for quick validation
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=1
    )
    val_iter = iter(val_loader)

    # Since we're streaming data, we need to estimate total steps
    warmup_steps = config.optim.warmup
    logger.info(f"Training will run for up to {total_steps} steps")
    logger.info(f"Warmup steps: {warmup_steps}")

    train_iter = iter(train_loader)

    model = EntropyModel(config).cuda()
    logger.info(f"Model: {model}")

    if getattr(config.optim, "enable_dynamo", False):
        logger.info("Torch Dynamo enabled: compiling model...")
        model = pt.compile(model)

    log_model_summary(model)
    param_count = count_parameters(model)
    logger.info(f"Model parameter count: {param_count}")
    wandb.run.summary["param_count"] = param_count

    optimizer = build_optimizer(model, config.optim)
    scheduler = get_cosine_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lr_min_ratio=config.optim.lr_min_ratio,
    )

    # Log initial generation before any training
    logger.info("Generating initial sample before training...")
    model.eval()
    prompt_tokens = pt.tensor([[tokenizer.bos_id]], dtype=pt.long).cuda()
    gen_tokens = generate_text_sample(
        model, prompt_tokens, max_new_tokens=256, temperature=1.0
    )
    response = tokenizer.decode(gen_tokens[0].tolist())

    val_table = wandb.Table(columns=["step", "response"])
    val_table.add_data(0, response)
    metrics = {
        "samples": val_table,
    }
    wandb.log(metrics, step=0)
    logger.info("Initial generation complete. Starting training...")
    model.train()

    if config.checkpoint.resume_from:
        if os.path.isfile(config.checkpoint.resume_from):
            checkpoint = pt.load(config.checkpoint.resume_from)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
            logger.info(f"Resumed training from checkpoint at step {step}.")

            skip_count = 0
            while skip_count < step:
                try:
                    _ = next(train_iter)
                    skip_count += 1
                except StopIteration:
                    train_iter = iter(train_loader)
            logger.info(
                f"Skipped {skip_count} batches to resume data loader at batch #{skip_count}."
            )
        else:
            logger.info(
                f"Checkpoint file {config.checkpoint.resume_from} not found. Starting fresh."
            )
            step = 0
    else:
        step = 0

    model.train()
    log_freq = config.logging.log_freq
    val_interval = config.training.val_interval
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_time = time.time()
    last_log_time = start_time

    interval_bytes = 0
    global_bytes = 0

    total_params = count_parameters(model)
    embed_params = config.model.vocab_size * config.model.dim
    non_embed_params = total_params - embed_params
    constant_flops_per_token = get_num_flop_per_token(
        non_embed_params,
        config.model.n_layers,
        config.model.dim,
        config.data.seq_len,
    )

    # Initialize loss tracking
    total_loss_sum = 0.0
    total_loss_count = 0

    while step < total_steps:
        t_fetch_start = time.time()
        try:
            input_seq, target_seq = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_seq, target_seq = next(train_iter)

        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()

        data_load_time = time.time() - t_fetch_start

        batch_bytes = input_seq.numel()
        interval_bytes += batch_bytes
        global_bytes += batch_bytes

        t_step_start = time.time()
        optimizer.zero_grad()
        logits = model(input_seq)
        loss = F.cross_entropy(
            logits.view(-1, config.model.vocab_size), target_seq.view(-1)
        )
        loss.backward()
        pt.nn.utils.clip_grad_norm_(model.parameters(), config.optim.clip)
        optimizer.step()
        scheduler.step()
        step += 1
        step_time = time.time() - t_step_start

        # Update loss tracking
        loss_val = loss.item()
        total_loss_sum += loss_val
        total_loss_count += 1
        avg_loss = total_loss_sum / total_loss_count

        if step % log_freq == 0:
            current_time = time.time()
            full_interval_time = current_time - last_log_time
            last_log_time = current_time

            current_lr = scheduler.get_last_lr()[0]
            grad_norm = (
                sum(
                    p.grad.data.norm(2).item()
                    for p in model.parameters()
                    if p.grad is not None
                )
            ) ** 0.5
            tokens_this_step = config.data.batch_size * config.data.seq_len
            tokens_processed = step * tokens_this_step
            wps = tokens_this_step / full_interval_time
            estimated_flops = tokens_processed * constant_flops_per_token
            bpb = loss_val / math.log(2)
            avg_bpb = avg_loss / math.log(2)

            elapsed_time = time.time() - start_time
            remaining_steps = total_steps - step
            avg_step_time = elapsed_time / step
            eta = remaining_steps * avg_step_time
            eta_str = str(datetime.timedelta(seconds=int(eta)))

            metrics = {
                "lr": current_lr,
                "loss": loss_val,
                "avg_loss": avg_loss,
                "bpb": bpb,
                "avg_bpb": avg_bpb,
                "wps": wps,
                "data_load_time": data_load_time,
                "step_time": step_time,
                "iter_time": full_interval_time,
                "eta": eta,
                "grad_norm": grad_norm,
                "estimated_flops": estimated_flops,
                "bytes": global_bytes,
            }
            logger.info(
                f"Step {step}: loss={loss_val:.4f} (avg={avg_loss:.4f}), "
                f"lr={current_lr:.2e}, grad_norm={grad_norm:.2e}, "
                f"iter_time={full_interval_time:.4f}s, wps={wps:.2e}, estimated_flops={estimated_flops:.2e}, "
                f"bytes={global_bytes:.2e}, bpb={bpb:.4f}, data_load_time={data_load_time:.4f}s, "
                f"ETA: {eta_str}"
            )
            log_metrics(metrics, step)

        if step % config.checkpoint.dump.every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                checkpoint_dir,
                config.checkpoint.dump.keep,
            )

        if step % val_interval == 0:
            model.eval()

            # Get next validation batch, recreate iterator if needed
            try:
                val_input, val_target = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_input, val_target = next(val_iter)

            # Compute validation loss on single batch
            val_input = val_input.cuda()
            val_target = val_target.cuda()
            with pt.no_grad():
                val_loss = F.cross_entropy(
                    model(val_input).view(-1, config.model.vocab_size),
                    val_target.view(-1),
                )
            val_bpb = val_loss.item() / math.log(2)

            # Generate sample starting with just BOS token
            val_table = wandb.Table(columns=["step", "response"])

            # Generate from BOS token only
            prompt_tokens = pt.tensor(
                [[tokenizer.bos_id]], dtype=pt.long
            ).cuda()  # BOS token
            gen_tokens = generate_text_sample(
                model, prompt_tokens, max_new_tokens=256, temperature=1.0
            )
            response = tokenizer.decode(gen_tokens[0].tolist())

            val_table.add_data(step, response)

            # Log validation metrics
            metrics = {
                "val_loss": val_loss.item(),
                "val_bpb": val_bpb,
                "samples": val_table,
            }
            log_metrics(metrics, step)
            logger.info(
                f"Validation step {step}: loss={val_loss.item():.4f}, bpb={val_bpb:.4f}"
            )

            model.train()

        if step >= total_steps:
            break

    final_elapsed = time.time() - start_time
    final_tokens = step * config.data.batch_size * config.data.seq_len
    final_estimated_flops = final_tokens * constant_flops_per_token

    logger.info(f"Training complete in {final_elapsed:.2f} seconds.")
    logger.info(f"Total tokens processed: {final_tokens}")
    logger.info(f"Total estimated FLOPs: {final_estimated_flops:.2e}")
    logger.info(f"Total bytes processed: {global_bytes}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
