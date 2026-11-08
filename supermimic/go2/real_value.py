"""Critic targets from real physics, stopping at resets and bootstrapping truncations."""

import torch

from .mdp import unflatten_state


@torch.no_grad()
def real_value_targets(learner, samples):
    mask = samples["mask"]
    lengths = mask.sum(-1).long()
    batch = torch.arange(len(mask), device=mask.device)
    final = lengths - 1
    initial = {key: value[:, 0] for key, value in samples.items() if key != "mask"}
    last = {key: value[batch, final] for key, value in samples.items() if key != "mask"}
    features = learner.critic_input(
        unflatten_state(initial["s"]), initial["a_prev"], initial["command"],
        initial["remaining"], initial["xi"], initial["terrain_id"], initial["route_heading"],
    )
    following = learner.critic_input(
        unflatten_state(last["s1"]), last["a"], last["command1"],
        last["remaining1"], last["xi"], last["terrain_id"], last["route_heading"],
    )
    discounts = learner.gamma ** torch.arange(mask.shape[1], device=mask.device)
    rewards = (samples["r"] * mask * discounts).sum(-1)
    bootstrap = learner.gamma**lengths * (~last["term"]).float() * learner.critic(following)
    return features, rewards + bootstrap, last["term"].float()


@torch.no_grad()
def validate_real_values(learner, retention):
    result = {}
    for name, samples in retention.real_validation.items():
        features, target, terminal = real_value_targets(learner, samples)
        error = learner.critic(features) - target
        falling = terminal.bool()
        result[name] = {
            "windows": len(features), "terminal_windows": int(falling.sum()),
            "td_rmse": error.square().mean().sqrt().item(), "value_bias": error.mean().item(),
            "terminal_td_rmse": error[falling].square().mean().sqrt().item() if falling.any() else None,
            "mean_real_target": target.mean().item(),
        }
    return result
