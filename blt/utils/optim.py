import math

import torch


def build_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def get_cosine_schedule(optimizer, warmup_steps, total_steps, lr_min_ratio):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / warmup_steps
        progress = float(current_step - warmup_steps) / (total_steps - warmup_steps)
        return max(lr_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
