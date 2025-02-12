import os
import argparse
import math
import time

import torch as pt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import wandb

from blt.model.entropy import EntropyModel
from blt.utils.optim import build_optimizer, get_cosine_schedule
from blt.utils.logging import init_wandb, log_metrics

# ----------------------------------------------------------------
# Simple Dataset: Reads a local text file (e.g., shakespeare.txt)
# and converts it into a stream of byte-level token IDs.
# (We reserve 1 for BOS and 2 for EOS; all byte values [0,255] are offset by 3.)
# ----------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, file_path, seq_len, bos_id=1, eos_id=2):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        byte_values = list(self.text.encode('utf-8'))
        self.token_ids = [b + 3 for b in byte_values]
        self.token_ids = [bos_id] + self.token_ids + [eos_id]
        self.seq_len = seq_len
        self.num_samples = (len(self.token_ids) - 1) // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # extra token for target shifting
        seq = self.token_ids[start:end]
        if len(seq) < self.seq_len + 1:
            seq += [0] * (self.seq_len + 1 - len(seq))
        input_seq = pt.tensor(seq[:-1], dtype=pt.long)
        target_seq = pt.tensor(seq[1:], dtype=pt.long)
        return input_seq, target_seq

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, keep):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.pt")
    pt.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    wandb.save(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Remove older checkpoints if the keep setting is positive.
    if keep > 0:
        all_ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")],
                           key=lambda x: int(x.split('_')[1].split('.')[0]))
        if len(all_ckpts) > keep:
            for ckpt in all_ckpts[:-keep]:
                os.remove(os.path.join(checkpoint_dir, ckpt))
                print(f"Removed old checkpoint {ckpt}")

def text_to_tokens(text, bos_id=1, eos_id=2):
    byte_values = list(text.encode('utf-8'))
    tokens = [b + 3 for b in byte_values]
    return [bos_id] + tokens  # Do not append EOS here for generation

def decode_tokens(tokens):
    token_list = tokens.tolist() if isinstance(tokens, pt.Tensor) else tokens
    filtered = [t - 3 for t in token_list if t not in (1, 2) and 0 <= t - 3 < 256]
    try:
        return bytes(filtered).decode('utf-8', errors='replace')
    except Exception:
        return ""

def generate_text_sample(model, prompt, max_new_tokens, temperature):
    """
    Autoregressively generate tokens using temperature sampling.
    For the first 128 generated tokens we force continuation (i.e. ignore EOS).
    """
    model.eval()
    if prompt.size(1) == 0:
        prompt = pt.tensor([[1]], dtype=pt.long).cuda()
    generated = prompt.clone()
    forced_tokens = 128  # force continuation for first 128 tokens
    with pt.no_grad():
        for i in range(max_new_tokens):
            logits = model(generated)  # [batch, cur_seq, vocab_size]
            next_logits = logits[0, -1]  # [vocab_size]
            scaled_logits = next_logits / temperature
            probabilities = pt.softmax(scaled_logits, dim=-1)
            next_token = pt.multinomial(probabilities, num_samples=1).unsqueeze(0)  # shape [1, 1]
            generated = pt.cat([generated, next_token], dim=1)
            # After forced_tokens, if we sample an EOS (id 2), then stop.
            if i >= forced_tokens and next_token.item() == 2:
                break
    model.train()
    return generated

# ----------------------------------------------------------------
# Main training loop (single GPU)
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train BLT Entropy Model on a text file")
    parser.add_argument("--config", type=str, default="blt/configs/entropy.yaml", help="Path to config YAML")
    parser.add_argument("overrides", nargs="*", help="Optional config overrides in dot list format")
    args = parser.parse_args()

    base_config = OmegaConf.load(args.config)
    if args.overrides:
        cli_conf = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(base_config, cli_conf)
    else:
        config = base_config

    # Determine checkpoint directory from config.
    checkpoint_dir = os.path.join(config.dump_dir, config.name)
    
    # Initialize wandb.
    init_wandb(config)
    wandb.run.name = config.name
    print(f"Run name: {config.name}")

    # Set random seed.
    pt.manual_seed(config.seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(config.seed)

    # Create dataset and dataloader.
    dataset = TextDataset(
        file_path=config.data.file,
        seq_len=config.data.seq_len,
        bos_id=1,
        eos_id=2
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.workers
    )

    # Initialize the entropy model.
    model = EntropyModel(config.model).cuda()

    # Count and log parameter count.
    param_count = count_parameters(model)
    print(f"Model parameter count: {param_count}")
    wandb.run.summary["param_count"] = param_count

    # Build optimizer and LR scheduler.
    optimizer = build_optimizer(model, config.optim)
    scheduler = get_cosine_schedule(
        optimizer,
        warmup_steps=config.optim.warmup,
        total_steps=config.training.total_steps,
        lr_min_ratio=config.optim.lr_min_ratio
    )

    model.train()
    step = 0
    log_freq = config.logging.log_freq
    val_interval = config.training.val_interval
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_time = time.time()

    # Define validation prompts.
    # The first prompt is empty so that the model generates freely.
    val_prompts = [""]
    temperatures = [0.5, 1.0, 1.5]
    max_new_tokens = 100

    while step < config.training.total_steps:
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.cuda()
            target_seq = target_seq.cuda()

            optimizer.zero_grad()
            logits = model(input_seq)  # [batch, seq_len, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, config.model.vocab_size),
                target_seq.view(-1)
            )
            loss.backward()
            pt.nn.utils.clip_grad_norm_(model.parameters(), config.optim.clip)
            optimizer.step()
            scheduler.step()
            step += 1

            # Logging training metrics.
            if step % log_freq == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                total_grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None
                ) ** 0.5
                tokens_processed = step * config.data.batch_size * config.data.seq_len
                wps = tokens_processed / elapsed
                iter_time = elapsed / step

                metrics = {
                    "loss": loss.item(),
                    "step": step,
                    "lr": current_lr,
                    "grad_norm": total_grad_norm,
                    "iter_time": iter_time,
                    "wps": wps,
                }
                print(f"Step {step}: loss={loss.item():.4f}, lr={current_lr:.2e}, "
                      f"grad_norm={total_grad_norm:.2e}, iter_time={iter_time:.4f} s, wps={wps:.2e}")
                log_metrics(metrics, step)

            # Save checkpoint every configured number of steps.
            if step % config.checkpoint.dump.every == 0:
                save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, config.checkpoint.dump.keep)

            # Validation: every val_interval steps, generate samples for each temperature.
            if step % val_interval == 0:
                sample_text_table = wandb.Table(columns=["id", "temperature", "response"])
                for i, temp in enumerate(temperatures):
                    for j, prompt in enumerate(val_prompts):
                        prompt_tokens = pt.tensor(text_to_tokens(prompt), dtype=pt.long).unsqueeze(0).cuda()
                        gen_tokens = generate_text_sample(model, prompt_tokens, max_new_tokens, temperature=temp)
                        response = decode_tokens(gen_tokens[0])
                        uid = f"{i}_{j}_{step}"
                        sample_text_table.add_data(uid, temp, response)
                wandb.log({"samples": sample_text_table})
                
            if step >= config.training.total_steps:
                break

    wandb.finish()

if __name__ == "__main__":
    main()
