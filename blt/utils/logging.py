import wandb


def init_wandb(config):
    run_id = config.logging.wandb_run_id
    if run_id is not None:
        run_id = run_id.strip()
    # If run_id is an empty string, set it to None.
    if not run_id:
        run_id = None

    # Initialize wandb without the id parameter if run_id is None.
    if run_id is None:
        wandb.init(project=config.logging.wandb_project, config=config, resume="allow")
    else:
        wandb.init(
            project=config.logging.wandb_project,
            config=config,
            id=run_id,
            resume="allow",
        )


def log_metrics(metrics, step):
    wandb.log(metrics, step=step)
