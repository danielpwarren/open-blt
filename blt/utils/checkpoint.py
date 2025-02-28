import logging
import os

import torch as pt
import wandb

logger = logging.getLogger()


def save_checkpoint(
    model, optimizer, scheduler, step, checkpoint_dir, scaler=None, keep=3
):
    """
    Saves a new checkpoint as 'checkpoint_latest.pt'. If a previous checkpoint_latest exists,
    it is renamed to include its step number (e.g. checkpoint_00500.pt). Then, only the last
    'keep' renamed checkpoints are retained.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    if os.path.exists(latest_path):
        try:
            prev_ckpt = pt.load(latest_path, map_location="cpu")
            prev_step = prev_ckpt.get("step", 0)
        except Exception as e:
            logger.info(
                "Warning: could not load previous checkpoint to get step; using 0", e
            )
            prev_step = 0
        renamed_path = os.path.join(checkpoint_dir, f"checkpoint_{prev_step:06d}.pt")
        os.rename(latest_path, renamed_path)
        logger.info(f"Renamed previous checkpoint to {renamed_path}")

    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if scaler is not None:
        checkpoint_data["scaler_state_dict"] = scaler.state_dict()

    pt.save(checkpoint_data, latest_path)
    wandb.log_artifact(latest_path, "checkpoint_latest.pt", "checkpoint")
    logger.info(f"Checkpoint saved at {latest_path}")

    all_ckpts = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_")
            and f.endswith(".pt")
            and f != "checkpoint_latest.pt"
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if len(all_ckpts) > keep:
        for ckpt in all_ckpts[:-keep]:
            os.remove(os.path.join(checkpoint_dir, ckpt))
            logger.info(f"Removed old checkpoint {ckpt}")
