import wandb


def init_wandb(config):
    wandb.init(project=config.logging.wandb_project, config=config)


def log_metrics(metrics, step):
    wandb.log(metrics, step=step)
