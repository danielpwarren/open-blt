import argparse
import datetime
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch as pt
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
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


@dataclass
class TrainingState:
    """Maintains the state of a training run."""
    model: pt.nn.Module
    optimizer: pt.optim.Optimizer
    scheduler: Any
    step: int
    scaler: Optional[GradScaler] = None
    total_loss_sum: float = 0.0
    total_loss_count: int = 0
    interval_bytes: int = 0
    global_bytes: int = 0
    start_time: float = 0.0
    last_log_time: float = 0.0
    
    @property
    def avg_loss(self) -> float:
        """Calculate average loss over all steps."""
        return self.total_loss_sum / max(1, self.total_loss_count)
    
    def update_loss_tracking(self, loss_val: float) -> None:
        """Update loss tracking statistics."""
        self.total_loss_sum += loss_val
        self.total_loss_count += 1
    
    def update_byte_tracking(self, batch_bytes: int) -> None:
        """Update byte tracking statistics."""
        self.interval_bytes += batch_bytes
        self.global_bytes += batch_bytes


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_text_sample(model, prompt, max_new_tokens, temperature, device):
    """
    Generate tokens autoregressively. For the first 128 tokens generated,
    force continuation (i.e. ignore EOS) so that a sample is produced.
    """
    model.eval()
    if prompt.size(1) == 0:
        prompt = pt.tensor([[257]], dtype=pt.long, device=device)
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
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False


def calculate_bpb(loss, bits_per_token=8):
    """Calculate bits per byte from loss, normalize by bit size."""
    return loss / math.log(2) / bits_per_token


def get_next_batch(train_iter, train_loader, device):
    """Get next batch from iterator, create new iterator if needed."""
    try:
        input_seq, target_seq = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        input_seq, target_seq = next(train_iter)
    
    return train_iter, input_seq.to(device), target_seq.to(device)


def get_validation_batch(val_iter, val_loader, device):
    """Get next validation batch, recreate iterator if needed."""
    try:
        val_input, val_target = next(val_iter)
    except StopIteration:
        val_iter = iter(val_loader)
        val_input, val_target = next(val_iter)
    
    return val_iter, val_input.to(device), val_target.to(device)


def setup_training_state(config, model, device, total_steps, resume_path=None):
    """Initialize or resume training state."""
    optimizer = build_optimizer(model, config.optim)
    scheduler = get_cosine_schedule(
        optimizer,
        warmup_steps=config.optim.warmup,
        total_steps=total_steps,
        lr_min_ratio=config.optim.lr_min_ratio,
    )
    
    use_amp = getattr(config.optim, "use_amp", False)
    scaler = GradScaler() if use_amp else None
    
    state = TrainingState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        scaler=scaler,
        start_time=time.time(),
        last_log_time=time.time()
    )
    
    if resume_path and os.path.isfile(resume_path):
        checkpoint = pt.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state.step = checkpoint["step"]
        if use_amp and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(f"Resumed training from checkpoint at step {state.step}.")
    
    return state


def log_training_progress(state, config, metrics, step_time, data_load_time, total_steps):
    """Log training progress with metrics."""
    current_time = time.time()
    full_interval_time = current_time - state.last_log_time
    state.last_log_time = current_time
    
    tokens_this_step = config.data.batch_size * config.data.seq_len
    tokens_processed = state.step * tokens_this_step
    wps = tokens_this_step / full_interval_time
    
    # Calculate ETA
    elapsed_time = current_time - state.start_time
    remaining_steps = total_steps - state.step
    avg_step_time = elapsed_time / state.step if state.step > 0 else 0
    eta = remaining_steps * avg_step_time
    eta_str = str(datetime.timedelta(seconds=int(eta)))
    
    # Add performance metrics
    metrics.update({
        "performance/wps": wps,
        "performance/data_load_time": data_load_time,
        "performance/step_time": step_time,
        "performance/iter_time": full_interval_time,
        "performance/eta": eta,
        "performance/bytes": state.global_bytes,
    })
    
    # Log to console
    logger_msg = (
        f"Step {state.step}: loss={metrics['train/loss']:.4f} "
        f"(avg={metrics['train/avg_loss']:.4f}), "
        f"lr={metrics['train/lr']:.2e}, grad_norm={metrics['train/grad_norm']:.2e}, "
        f"iter_time={full_interval_time:.4f}s, wps={wps:.2e}, "
        f"bytes={state.global_bytes:.2e}, bpb={metrics['train/bpb']:.4f}, "
        f"data_load_time={data_load_time:.4f}s, ETA: {eta_str}"
    )
    logger.info(logger_msg)
    log_metrics(metrics, state.step)


def perform_training_step(state, input_seq, target_seq, config, device):
    """Perform a single training step."""
    t_step_start = time.time()
    state.optimizer.zero_grad()
    
    use_amp = state.scaler is not None
    
    # Use autocast for mixed precision training
    if use_amp:
        with autocast(device_type=device if device != "cpu" else None):
            logits = state.model(input_seq)
            loss = F.cross_entropy(
                logits.view(-1, config.model.vocab_size), target_seq.view(-1)
            )

        state.scaler.scale(loss).backward()
        state.scaler.unscale_(state.optimizer)
        pt.nn.utils.clip_grad_norm_(state.model.parameters(), config.optim.clip)
        state.scaler.step(state.optimizer)
        state.scaler.update()
    else:
        logits = state.model(input_seq)
        loss = F.cross_entropy(
            logits.view(-1, config.model.vocab_size), target_seq.view(-1)
        )
        loss.backward()
        pt.nn.utils.clip_grad_norm_(state.model.parameters(), config.optim.clip)
        state.optimizer.step()
    
    state.scheduler.step()
    state.step += 1
    step_time = time.time() - t_step_start
    
    # Update loss tracking
    loss_val = loss.item()
    state.update_loss_tracking(loss_val)
    
    # Calculate gradient norm
    grad_norm = (
        sum(
            p.grad.data.norm(2).item()
            for p in state.model.parameters()
            if p.grad is not None
        )
    ) ** 0.5
    
    return loss_val, step_time, grad_norm


def validate_model(state, val_loader, device, config, tokenizer):
    """Run validation on the entire validation set and generate sample text."""
    state.model.eval()
    
    # Validate on entire validation set
    total_val_loss = 0.0
    total_val_samples = 0
    use_amp = state.scaler is not None
    
    with pt.no_grad():
        for val_input, val_target in val_loader:
            val_input = val_input.to(device)
            val_target = val_target.to(device)
            
            if use_amp:
                with autocast(device_type=device if device != "cpu" else None):
                    val_logits = state.model(val_input)
                    val_loss = F.cross_entropy(
                        val_logits.view(-1, config.model.vocab_size),
                        val_target.view(-1),
                    )
            else:
                val_logits = state.model(val_input)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, config.model.vocab_size),
                    val_target.view(-1),
                )
            
            batch_size = val_input.size(0)
            total_val_loss += val_loss.item() * batch_size
            total_val_samples += batch_size
    
    # Calculate average loss across all validation samples
    avg_val_loss = total_val_loss / total_val_samples
    val_bpb = calculate_bpb(avg_val_loss)
    
    # Generate sample starting with just BOS token
    val_table = wandb.Table(columns=["step", "response"])
    
    # Generate from BOS token only
    prompt_tokens = pt.tensor([[tokenizer.bos_id]], dtype=pt.long, device=device)
    gen_tokens = generate_text_sample(
        state.model, prompt_tokens, max_new_tokens=256, temperature=1.0, device=device
    )
    response = tokenizer.decode(gen_tokens[0].tolist())
    
    val_table.add_data(state.step, response)
    
    # Log validation metrics
    metrics = {
        "validation/loss": avg_val_loss,
        "validation/val_bpb": val_bpb,
        "samples": val_table,
    }
    log_metrics(metrics, state.step)
    logger.info(
        f"Validation step {state.step}: loss={avg_val_loss:.4f}, bpb={val_bpb:.4f} (full validation set)"
    )
    
    state.model.train()
    return avg_val_loss


def train_model(config, model, train_loader, val_loader, tokenizer, checkpoint_dir, total_steps, device):
    """Main training loop with modular components."""
    # Initialize training state
    state = setup_training_state(
        config, 
        model, 
        device, 
        total_steps, 
        resume_path=config.checkpoint.resume_from if hasattr(config.checkpoint, "resume_from") else None
    )
    
    # Skip batches if resuming
    train_iter = iter(train_loader)
    if state.step > 0:
        skip_count = 0
        while skip_count < state.step:
            try:
                _ = next(train_iter)
                skip_count += 1
            except StopIteration:
                train_iter = iter(train_loader)
        logger.info(f"Skipped {skip_count} batches to resume at batch #{skip_count}.")
    
    # Configuration values
    log_freq = config.logging.log_freq
    val_interval = config.training.val_interval
    
    # Calculate model constants for metrics
    total_params = count_parameters(model)
    embed_params = config.model.vocab_size * config.model.dim
    non_embed_params = total_params - embed_params
    constant_flops_per_token = get_num_flop_per_token(
        non_embed_params,
        config.model.n_layers,
        config.model.dim,
        config.data.seq_len,
    )
    
    # Main training loop
    while state.step < total_steps:
        # Fetch batch
        t_fetch_start = time.time()
        train_iter, input_seq, target_seq = get_next_batch(train_iter, train_loader, device)
        data_load_time = time.time() - t_fetch_start
        
        # Track batch size in bytes
        batch_bytes = input_seq.numel()
        state.update_byte_tracking(batch_bytes)
        
        # Perform training step
        loss_val, step_time, grad_norm = perform_training_step(
            state, input_seq, target_seq, config, device
        )
        
        # Log progress periodically
        if state.step % log_freq == 0:
            current_lr = state.scheduler.get_last_lr()[0]
            bpb = calculate_bpb(loss_val)
            avg_bpb = calculate_bpb(state.avg_loss)
            
            # Calculate tokens and estimated FLOPs
            tokens_processed = state.step * config.data.batch_size * config.data.seq_len
            estimated_flops = tokens_processed * constant_flops_per_token
            
            metrics = {
                "train/grad_norm": grad_norm,
                "train/loss": loss_val,
                "train/bpb": bpb,
                "train/avg_loss": state.avg_loss,
                "train/avg_bpb": avg_bpb,
                "train/lr": current_lr,
                "performance/estimated_flops": estimated_flops,
            }
            
            log_training_progress(state, config, metrics, step_time, data_load_time, total_steps)
        
        # Save checkpoint periodically
        if state.step % config.checkpoint.dump.every == 0:
            save_checkpoint(
                state.model,
                state.optimizer,
                state.scheduler,
                state.step,
                checkpoint_dir,
                state.scaler,
                config.checkpoint.dump.keep,
            )
        
        # Run validation periodically
        if state.step % val_interval == 0:
            validate_model(state, val_loader, device, config, tokenizer)
    
    # Log final statistics
    final_elapsed = time.time() - state.start_time
    final_tokens = state.step * config.data.batch_size * config.data.seq_len
    final_estimated_flops = final_tokens * constant_flops_per_token
    
    logger.info(f"Training complete in {final_elapsed:.2f} seconds.")
    logger.info(f"Total tokens processed: {final_tokens}")
    logger.info(f"Total estimated FLOPs: {final_estimated_flops:.2e}")
    logger.info(f"Total bytes processed: {state.global_bytes}")
    
    return state


def calculate_total_steps(config, train_dataset):
    """Calculate or retrieve cached total steps."""
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
                return total_steps
            else:
                logger.info("Config changed, recalculating total steps...")
    else:
        logger.info("Calculating total steps for the first time...")
    
    total_steps = train_dataset.calculate_total_steps(config.data.batch_size)
    os.makedirs(config.dump_dir, exist_ok=True)
    with open(steps_cache_file, "w") as f:
        json.dump({"config": cache_config, "total_steps": total_steps}, f)
    
    return total_steps


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
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (e.g., 'cuda', 'cpu', 'mps'). Empty string means auto-detect.",
    )
    args = parser.parse_args()

    # Set up device
    if args.device:
        device = args.device
    else:
        device = "cuda" if pt.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")

    # Load and merge configuration
    base_config = OmegaConf.load(args.config)
    config = (
        OmegaConf.merge(base_config, OmegaConf.from_dotlist(args.overrides))
        if args.overrides
        else base_config
    )

    # Set up logging
    log_dir = os.path.join("logs", os.path.basename(args.config).split(".")[0])
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file=log_file)
    logger.info(f"Full Config: {config}")

    # Initialize directories and wandb
    checkpoint_dir = os.path.join(config.dump_dir, config.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    init_wandb(config)
    logger.info(f"Run name: {config.name}")

    # Set random seed
    seed_everything(config.seed)

    # Initialize tokenizer
    tokenizer = ByteTokenizer()

    # Create datasets
    train_dataset = JsonlDataset(
        data_dir=config.data.train_dir,
        seq_len=config.data.seq_len,
        shuffle_files=True,
        seed=config.seed,
    )
    
    val_dataset = JsonlValidationDataset(
        val_file=config.data.val_file,
        seq_len=config.data.seq_len,
        max_samples=config.data.max_val_samples,
    )
    
    # Calculate total steps
    total_steps = calculate_total_steps(config, train_dataset)
    logger.info(f"Total training steps: {total_steps}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
    )
    
    val_batch_size = 8  # Small fixed batch size for quick validation
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=1
    )

    # Create model
    model = EntropyModel(config).to(device)
    logger.info(f"Model: {model}")
    
    # Enable dynamo compilation if configured
    if getattr(config.optim, "enable_dynamo", False):
        logger.info("Torch Dynamo enabled: compiling model...")
        model = pt.compile(model)
    
    # Log model summary
    log_model_summary(model)
    param_count = count_parameters(model)
    logger.info(f"Model parameter count: {param_count}")
    wandb.run.summary["param_count"] = param_count
    
    # Log initial generation before training
    logger.info("Generating initial sample before training...")
    model.eval()
    prompt_tokens = pt.tensor([[tokenizer.bos_id]], dtype=pt.long, device=device)
    gen_tokens = generate_text_sample(
        model, prompt_tokens, max_new_tokens=256, temperature=1.0, device=device
    )
    response = tokenizer.decode(gen_tokens[0].tolist())
    
    val_table = wandb.Table(columns=["step", "response"])
    val_table.add_data(0, response)
    metrics = {"samples": val_table}
    wandb.log(metrics, step=0)
    logger.info("Initial generation complete. Starting training...")
    model.train()
    
    # Train model
    train_model(config, model, train_loader, val_loader, tokenizer, checkpoint_dir, total_steps, device)
    
    # Clean up
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()