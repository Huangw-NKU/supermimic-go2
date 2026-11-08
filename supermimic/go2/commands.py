"""Command scheduling, including the same timer transitions in imagined rollouts."""

import torch

from .config import CommandCfg


def sample_commands(count, device, cfg: CommandCfg, scale=1.0):
    bounds = torch.tensor((cfg.vx, cfg.vy, cfg.yaw), device=device)
    command = (
        bounds[:, 0]
        + torch.rand(count, 3, device=device) * (bounds[:, 1] - bounds[:, 0])
    ) * scale
    command[torch.rand(count, device=device) < cfg.standing_probability] = 0
    duration = torch.randint(cfg.min_steps, cfg.max_steps + 1, (count,), device=device)
    return command, duration


def advance_commands(command, remaining, cfg: CommandCfg, scale=1.0, fixed=None):
    remaining = remaining - (1 if fixed is None else (~fixed).long())
    sampled, duration = sample_commands(len(command), command.device, cfg, scale)
    due = remaining <= 0
    return torch.where(due[:, None], sampled, command), torch.where(
        due, duration, remaining
    )
