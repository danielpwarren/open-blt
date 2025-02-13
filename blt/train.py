import argparse
import math
import os
import time

import torch as pt
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import wandb
from blt.data.binary_dataset import BinaryDataset
from blt.model.entropy import EntropyModel
from blt.model.utils import get_num_flop_per_token
from blt.utils.logging import init_wandb, log_metrics
from blt.utils.optim import build_optimizer, get_cosine_schedule


# -------------------------------
# Helper functions
# -------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, keep):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.pt")
    pt.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    wandb.save(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    # Remove older checkpoints if the number saved exceeds 'keep'
    if keep > 0:
        all_ckpts = sorted(
            [
                f
                for f in os.listdir(checkpoint_dir)
                if f.startswith("checkpoint_") and f.endswith(".pt")
            ],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if len(all_ckpts) > keep:
            for ckpt in all_ckpts[:-keep]:
                os.remove(os.path.join(checkpoint_dir, ckpt))
                print(f"Removed old checkpoint {ckpt}")


# For generation, we update text_to_tokens to leave byte values unchanged.
def text_to_tokens(text, bos_id=257, eos_id=258):
    token_list = list(text.encode("utf-8"))
    return [bos_id] + token_list  # (EOS is not appended for generation)


def decode_tokens(tokens):
    token_list = tokens.tolist() if isinstance(tokens, pt.Tensor) else tokens
    # Remove our special tokens (bos=257, eos=258) before decoding.
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
    forced_tokens = 128  # force continuation for the first 128 tokens
    with pt.no_grad():
        for i in range(max_new_tokens):
            logits = model(generated)  # shape: [1, cur_seq, vocab_size]
            next_logits = logits[0, -1]  # last token's logits
            scaled_logits = next_logits / temperature
            probabilities = pt.softmax(scaled_logits, dim=-1)
            next_token = pt.multinomial(probabilities, num_samples=1).unsqueeze(0)
            generated = pt.cat([generated, next_token], dim=1)
            if i >= forced_tokens and next_token.item() == 258:
                break
    model.train()
    return generated


# -------------------------------
# Main training loop
# -------------------------------
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

    checkpoint_dir = os.path.join(config.dump_dir, config.name)

    init_wandb(config)
    wandb.run.name = config.name
    print(f"Run name: {config.name}")

    pt.manual_seed(config.seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(config.seed)

    dataset = BinaryDataset(file_path=config.data.file, seq_len=config.data.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.workers,
    )
    model = EntropyModel(config.model).cuda()

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

    model.train()
    step = 0
    log_freq = config.logging.log_freq
    val_interval = config.training.val_interval
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_time = time.time()

    # Variables to accumulate metrics over the logging interval.
    interval_data_load_time = 0.0
    interval_bytes = 0
    global_bytes = 0  # cumulative bytes processed

    # Precompute constant FLOPs per token (using non-embedding parameters)
    total_params = count_parameters(model)
    embed_params = config.model.vocab_size * config.model.dim
    non_embed_params = total_params - embed_params
    constant_flops_per_token = get_num_flop_per_token(
        non_embed_params,
        config.model.n_layers,
        config.model.dim,
        config.data.seq_len,
    )

    val_prompts = [""]  # empty prompt for free generation.
    temperatures = [0.5, 1.0, 1.5]
    max_new_tokens = 256

    while step < config.training.total_steps:
        for input_seq, target_seq in dataloader:
            # Measure data loading time.
            data_load_start = time.time()
            input_seq = input_seq.cuda()
            target_seq = target_seq.cuda()
            data_load_time = time.time() - data_load_start
            interval_data_load_time += data_load_time

            # Count bytes processed (each token is stored as an integer)
            batch_bytes = input_seq.numel()
            interval_bytes += batch_bytes
            global_bytes += batch_bytes

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

            if step % log_freq == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                grad_norm = (
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    )
                ) ** 0.5
                tokens_processed = step * config.data.batch_size * config.data.seq_len
                wps = tokens_processed / elapsed

                # Compute cumulative estimated FLOPs so far.
                global_estimated_flops = tokens_processed * constant_flops_per_token

                bpb = loss.item() / math.log(2)
                metrics = {
                    "loss": loss.item(),
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "iter_time": elapsed / step,
                    "wps": wps,
                    "global_estimated_flops": global_estimated_flops,
                    "global_bytes": global_bytes,
                    "data_load_time": interval_data_load_time / log_freq,
                    "bpb": bpb,
                }
                print(
                    f"Step {step}: loss={loss.item():.4f}, lr={current_lr:.2e}, grad_norm={grad_norm:.2e}, "
                    f"iter_time={metrics['iter_time']:.4f} s, wps={wps:.2e}, "
                    f"global_estimated_flops={global_estimated_flops:.2e}, global_bytes={global_bytes:.2e}, "
                    f"bpb={bpb:.4f}, data_load_time={metrics['data_load_time']:.4f} s"
                )
                log_metrics(metrics, step)
                # Reset the interval accumulators.
                interval_data_load_time = 0.0
                interval_bytes = 0

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
                sample_text_table = wandb.Table(
                    columns=["id", "temperature", "token_ids", "response"]
                )
                for i, temp in enumerate(temperatures):
                    for j, prompt in enumerate(val_prompts):
                        prompt_tokens = (
                            pt.tensor(text_to_tokens(prompt), dtype=pt.long)
                            .unsqueeze(0)
                            .cuda()
                        )
                        gen_tokens = generate_text_sample(
                            model, prompt_tokens, max_new_tokens, temperature=temp
                        )
                        response = decode_tokens(gen_tokens[0])
                        uid = f"{i}_{j}_{step}"
                        sample_text_table.add_data(
                            uid, temp, gen_tokens[0].tolist(), response
                        )
                wandb.log({"samples": sample_text_table})

            if step >= config.training.total_steps:
                break

    # Log final cumulative metrics.
    final_elapsed = time.time() - start_time
    final_tokens = step * config.data.batch_size * config.data.seq_len
    final_estimated_flops = final_tokens * constant_flops_per_token
    print(f"Training complete in {final_elapsed:.2f} seconds.")
    print(f"Total tokens processed: {final_tokens}")
    print(f"Total estimated FLOPs: {final_estimated_flops:.2e}")
    print(f"Total bytes processed: {global_bytes}")

    wandb.log(
        {
            "final_elapsed": final_elapsed,
            "final_tokens": final_tokens,
            "final_estimated_flops": final_estimated_flops,
            "global_bytes": global_bytes,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
