import logging
import math
import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict, Optional

import wandb


class LogFormatter(logging.Formatter):

    def __init__(self):
        self.start_time = time.time()

    def formatTime(self, record):
        subsecond, seconds = math.modf(record.created)
        curr_date = (
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds))
            + f".{int(subsecond * 1_000_000):06d}"
        )
        delta = timedelta(seconds=round(record.created - self.start_time))
        return f"{curr_date} - {delta}"

    def formatPrefix(self, record):
        fmt_time = self.formatTime(record)
        return f"{record.levelname:<7} {fmt_time} - "

    def formatMessage(self, record, indent: str):
        content = record.getMessage()
        content = content.replace("\n", "\n" + indent)

        # Exception handling
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            content = content + indent.join(
                [l + "\n" for l in record.exc_text.splitlines()]
            )
            if content[-1:] == "\n":
                content = content[:-1]
        if record.stack_info:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            stack_text = self.formatStack(record.stack_info)
            content = content + indent.join([l + "\n" for l in stack_text.splitlines()])
            if content[-1:] == "\n":
                content = content[:-1]

        return content

    def format(self, record):
        prefix = self.formatPrefix(record)
        indent = " " * len(prefix)
        content = self.formatMessage(record, indent)
        return prefix + content


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up logging to both console and file (if specified).

    Args:
        log_file: Path to log file. If None, logging to file is disabled.
        level: Logging level (default: INFO)
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Create formatter
    formatter = LogFormatter()

    # Create console handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Add stderr handler for warnings and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    logger.info("Logging setup complete")


def init_wandb(config: Any, resume: bool = False) -> None:
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        config: Configuration object or dictionary
        resume: Whether to resume a previous run
    """
    if not hasattr(config.logging, "wandb_project"):
        logging.warning(
            "WandB project not specified in config, skipping WandB initialization"
        )
        return

    config_dict = config.__dict__ if hasattr(config, "__dict__") else config

    # Check for run_id for resuming
    run_id = getattr(config, "wandb_run_id", None)
    if run_id is not None and run_id.strip() == "":
        run_id = None

    wandb.init(
        project=config.logging.wandb_project,
        entity=getattr(config.logging, "wandb_entity", None),
        config=config_dict,
        name=getattr(config, "name", None),
        id=run_id,
        resume="allow",
    )

    logging.info(f"WandB initialized: {wandb.run.name}")


def log_metrics(metrics: Dict[str, Any], step: int) -> None:
    """
    Log metrics to WandB (if initialized).

    Args:
        metrics: Dictionary of metrics to log
        step: Current step/iteration
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_artifact(
    artifact_path: str,
    name: str,
    artifact_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an artifact (file) to WandB.

    Args:
        artifact_path: Path to the artifact file
        name: Name of the artifact
        artifact_type: Type of artifact (e.g., 'model', 'dataset')
        metadata: Additional metadata to associate with the artifact
    """
    if wandb.run is None:
        logging.warning("WandB not initialized, skipping artifact logging")
        return

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
    artifact.add_file(artifact_path)
    wandb.log_artifact(artifact)

    logging.info(f"Artifact logged to WandB: {name} ({artifact_type})")


def log_model_summary(model, input_size=None) -> None:
    """
    Log model summary to WandB.

    Args:
        model: PyTorch model
        input_size: Input size for model summary (optional)
    """
    if wandb.run is None:
        logging.warning("WandB not initialized, skipping model summary logging")
        return

    try:
        wandb.watch(model, log="all")
        logging.info("Model summary logged to WandB")
    except Exception as e:
        logging.error(f"Failed to log model summary to WandB: {e}")
