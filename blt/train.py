import argparse
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

from blt.data.binary_dataset import BinaryDataset
from blt.model.entropy import EntropyModel
from blt.model.utils import get_num_flop_per_token
from blt.utils.logging import init_wandb, log_metrics
from blt.utils.optim import build_optimizer, get_cosine_schedule


# Helper functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, keep=3):
    """
    Saves a new checkpoint as 'checkpoint_latest.pt'. If a previous checkpoint_latest exists,
    it is renamed to include its step number (e.g. checkpoint_00500.pt). Then, only the last
    'keep' renamed checkpoints are retained.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    # If a latest checkpoint exists, rename it to its step-numbered filename.
    if os.path.exists(latest_path):
        try:
            prev_ckpt = pt.load(latest_path, map_location="cpu")
            prev_step = prev_ckpt.get("step", 0)
        except Exception as e:
            print("Warning: could not load previous checkpoint to get step; using 0", e)
            prev_step = 0
        renamed_path = os.path.join(checkpoint_dir, f"checkpoint_{prev_step:06d}.pt")
        os.rename(latest_path, renamed_path)
        print(f"Renamed previous checkpoint to {renamed_path}")

    # Save the new checkpoint as 'checkpoint_latest.pt'
    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    pt.save(checkpoint_data, latest_path)
    wandb.save(latest_path)
    print(f"Checkpoint saved at {latest_path}")

    # Remove older checkpoints if more than 'keep' exist.
    # We consider only files named "checkpoint_{step}.pt" (excluding checkpoint_latest.pt).
    all_ckpts = sorted(
        [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt") and f != "checkpoint_latest.pt"
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if len(all_ckpts) > keep:
        for ckpt in all_ckpts[:-keep]:
            os.remove(os.path.join(checkpoint_dir, ckpt))
            print(f"Removed old checkpoint {ckpt}")


def text_to_tokens(text, bos_id=257, eos_id=258):
    """
    Convert text into a list of raw byte tokens, prepended by BOS.
    Note: We do NOT append EOS for generation prompts.
    """
    token_list = list(text.encode("utf-8"))
    return [bos_id] + token_list


def decode_tokens(tokens):
    """
    Convert a list of tokens (ints) back to UTF-8 text, omitting BOS/EOS.
    """
    token_list = tokens.tolist() if isinstance(tokens, pt.Tensor) else tokens
    filtered = [t for t in token_list if t not in (257, 258)]
    try:
        return bytes(filtered).decode("utf-8", errors="replace")
    except Exception:
        return ""


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
    """
    Ensures reproducible behavior by setting seeds for python, numpy, and torch.
    Disables CUDA benchmarking to ensure deterministic behavior where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False


# Main training loop
def main():
    parser = argparse.ArgumentParser(
        description="Train BLT Entropy Model on a binary token file"
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
    print(config)

    checkpoint_dir = os.path.join(config.dump_dir, config.name)

    init_wandb(config)
    wandb.run.name = config.name
    print(f"Run name: {config.name}")

    seed_everything(config.seed)

    dataset = BinaryDataset(file_path=config.data.file, seq_len=config.data.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
    )

    # Use a manual iterator to log loading times
    dataloader_iter = iter(dataloader)

    model = EntropyModel(config).cuda()
    print(model)

    if getattr(config.optim, "enable_dynamo", False):
        print("Torch Dynamo enabled: compiling model...")
        model = pt.compile(model, backend="inductor", mode="max-autotune")

    wandb.watch(model, log="all")
    param_count = count_parameters(model)
    print(f"Model parameter count: {param_count}")
    wandb.run.summary["param_count"] = param_count

    optimizer = build_optimizer(model, config.optim)
    scheduler = get_cosine_schedule(
        optimizer,
        warmup_steps=config.optim.warmup,
        total_steps=config.training.total_steps,
        lr_min_ratio=config.optim.lr_min_ratio,
    )

    # Resume from checkpoint if provided
    if config.checkpoint.resume_from:
        if os.path.isfile(config.checkpoint.resume_from):
            checkpoint = pt.load(config.checkpoint.resume_from)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
            print(f"Resumed training from checkpoint at step {step}.")

            skip_count = 0
            while skip_count < step:
                try:
                    _ = next(dataloader_iter)
                    skip_count += 1
                except StopIteration:
                    dataloader_iter = iter(dataloader)
            print(
                f"Skipped {skip_count} batches to resume data loader at batch #{skip_count}."
            )
        else:
            print(
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

    # Variables to accumulate metrics
    interval_bytes = 0
    global_bytes = 0

    # Precompute FLOPs per token
    total_params = count_parameters(model)
    embed_params = config.model.vocab_size * config.model.dim
    non_embed_params = total_params - embed_params
    constant_flops_per_token = get_num_flop_per_token(
        non_embed_params,
        config.model.n_layers,
        config.model.dim,
        config.data.seq_len,
    )

    # Two validation prompts: one empty and one with sample text.
    val_prompts = ["", "The meaning of life is", "Game of Thrones is", "The best way to cook pasta is"]
    temperatures = [0.5, 0.75, 1.0, 1.25, 1.5]
    max_new_tokens = 256

    while step < config.training.total_steps:
        # Capture full fetch time (from DataLoader)
        t_fetch_start = time.time()
        try:
            input_seq, target_seq = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_seq, target_seq = next(dataloader_iter)

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

        # Log metrics for this step
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
            bpb = loss.item() / math.log(2)

            # Compute overall elapsed time and ETA
            elapsed_time = time.time() - start_time
            remaining_steps = config.training.total_steps - step
            avg_step_time = elapsed_time / step
            eta = remaining_steps * avg_step_time

            metrics = {
                "lr": current_lr,
                "loss": loss.item(),
                "bpb": bpb,
                "wps": wps,
                "data_load_time": data_load_time,
                "step_time": step_time,
                "iter_time": full_interval_time,
                "eta": eta,
                "grad_norm": grad_norm,
                "estimated_flops": estimated_flops,
                "bytes": global_bytes,
            }
            print(
                f"Step {step}: loss={loss.item():.4f}, lr={current_lr:.2e}, grad_norm={grad_norm:.2e}, "
                f"iter_time={full_interval_time:.4f}s, wps={wps:.2e}, estimated_flops={estimated_flops:.2e}, "
                f"bytes={global_bytes:.2e}, bpb={bpb:.4f}, data_load_time={data_load_time:.4f}s, "
                f"ETA: {eta:.1f}s"
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
            # Create a new table for this validation interval
            val_table = wandb.Table(
                columns=["step", "id", "temperature", "prompt", "token_ids", "response"]
            )
            for prompt in val_prompts:
                for i, temp in enumerate(temperatures):
                    prompt_tokens = (
                        pt.tensor(text_to_tokens(prompt), dtype=pt.long)
                        .unsqueeze(0)
                        .cuda()
                    )
                    gen_tokens = generate_text_sample(
                        model, prompt_tokens, max_new_tokens, temperature=temp
                    )
                    response = decode_tokens(gen_tokens[0])
                    uid = f"{hash(prompt)}_{i}_{step}"
                    val_table.add_data(
                        step, uid, temp, prompt, gen_tokens[0].tolist(), response
                    )
            wandb.log({"samples": val_table}, step=step)

        if step >= config.training.total_steps:
            break

    final_elapsed = time.time() - start_time
    final_tokens = step * config.data.batch_size * config.data.seq_len
    final_estimated_flops = final_tokens * constant_flops_per_token

    print(f"Training complete in {final_elapsed:.2f} seconds.")
    print(f"Total tokens processed: {final_tokens}")
    print(f"Total estimated FLOPs: {final_estimated_flops:.2e}")
    print(f"Total bytes processed: {global_bytes}")

    wandb.finish()


if __name__ == "__main__":
    main()
