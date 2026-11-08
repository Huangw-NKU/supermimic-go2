"""Frozen successful-policy examples and fixed dynamics validation windows."""

import math

import torch

from .config import NUM_FEET, NUM_J, OBS_DIM
from .mdp import normalize_xi, unflatten_state


RETENTION_GROUPS = ("flat", "stairs_down", "boxes", "stairs_up_lower")
VALIDATION_GROUPS = (*RETENTION_GROUPS, "stairs_up_l5")
RETENTION_GROUPS_V2 = (*RETENTION_GROUPS, "stairs_up_l5", "stairs_down_l5")
RETENTION_WEIGHTS_V2 = (0.25, 0.125, 0.25, 0.125, 0.125, 0.125)


class RetentionData:
    def __init__(self, path, device, checkpoint):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") not in ("supermimic-go2-retention-v1", "supermimic-go2-retention-v2"):
            raise ValueError("Expected a Go2 successful-trajectory retention dataset")
        for key in ("joint_names", "network_cfg", "task_cfg"):
            if data[key] != checkpoint[key]:
                raise ValueError(f"Retention data and checkpoint differ in {key}")
        history = checkpoint["network_cfg"]["hist"]
        modern = data["format"].endswith("v2")
        self.names = RETENTION_GROUPS_V2 if modern else RETENTION_GROUPS
        self.weights = RETENTION_WEIGHTS_V2 if modern else (0.25,) * 4
        self.groups, self.guard_groups = [], []
        for name in self.names:
            group = data["policy"][name]
            count = len(group["history"])
            expected = {
                "history": (count, history, OBS_DIM + NUM_FEET),
                "command": (count, 3),
                "action": (count, NUM_J),
            }
            if count == 0 or any(group[key].shape != shape for key, shape in expected.items()):
                raise ValueError(f"Invalid retention example shapes in {name}")
            if any(not torch.isfinite(group[key]).all() for key in expected):
                raise ValueError(f"Nonfinite retention examples in {name}")
            self.groups.append({key: group[key].to(device) for key in expected})
            if modern:
                guard = data["guard"][name]
                count = len(guard["history"])
                shapes = {"history": (count, history, OBS_DIM + NUM_FEET), "command": (count, 3), "action": (count, NUM_J)}
                if count < 128 or any(guard[key].shape != shape for key, shape in shapes.items()):
                    raise ValueError(f"Invalid fixed-guard histories in {name}")
                if any(not torch.isfinite(guard[key]).all() for key in shapes):
                    raise ValueError(f"Nonfinite fixed-guard histories in {name}")
                selection = torch.linspace(0, count - 1, 128).round().long()
                self.guard_groups.append({key: guard[key][selection].to(device) for key in shapes})
        validation_names = self.names if modern else VALIDATION_GROUPS
        self.validation = {
            name: {key: value.to(device) for key, value in data["validation"][name].items()}
            for name in validation_names
        }
        self.real_validation = {
            name: {key: value.to(device) for key, value in group.items()}
            for name, group in data.get("real_validation", {}).items()
        }
        self.teacher_sha256 = data["teacher_sha256"]

    def sample(self, batch):
        parts = []
        for count, group in zip(self.counts(batch), self.groups, strict=True):
            ids = torch.randint(len(group["history"]), (count,), device=group["history"].device)
            parts.append({key: value[ids] for key, value in group.items()})
        return tuple(
            torch.cat([part[key] for part in parts], dim=0)
            for key in ("history", "command", "action")
        )

    def counts(self, batch):
        exact = [batch * weight for weight in self.weights]
        counts = [int(x) for x in exact]
        order = sorted(range(len(counts)), key=lambda i: (-(exact[i] - counts[i]), i))
        for index in order[:batch - sum(counts)]:
            counts[index] += 1
        return counts

    def group_mse(self, predicted, target):
        counts = self.counts(len(predicted))
        return torch.stack([
            (p - t).square().mean()
            for p, t in zip(predicted.split(counts), target.split(counts), strict=True)
            if len(p)
        ])

    @torch.no_grad()
    def guard_mse(self, policy):
        if not self.guard_groups:
            raise ValueError("Action-budget guarding requires a v2 reference bank")
        history, command, target = (
            torch.cat([g[key] for g in self.guard_groups])
            for key in ("history", "command", "action")
        )
        squared = (policy(history, command) - target).square().mean(-1)
        return squared.view(len(self.guard_groups), -1).mean(-1)

    @torch.no_grad()
    def measure_policy(self, policy):
        result = {}
        for name, group in zip(self.names, self.groups, strict=True):
            errors = torch.cat([
                (policy(h, c) - a).square().mean(-1)
                for h, c, a in zip(group["history"].split(1024), group["command"].split(1024), group["action"].split(1024), strict=True)
            ])
            result[name] = {"examples": len(errors), "rmse": errors.mean().sqrt().item(), "p95_example_rmse": errors.sqrt().quantile(0.95).item()}
        if self.guard_groups:
            for name, value in zip(self.names, self.guard_mse(policy), strict=True):
                result[name]["fixed_guard_rmse"] = value.sqrt().item()
        return result

    @torch.no_grad()
    def validate(self, model, terrain):
        result = {}
        for name, group in self.validation.items():
            state = unflatten_state(group["s0"])
            errors, contact_errors = [], []
            joint_errors = []
            for step in range(group["a"].shape[1]):
                ground, support = terrain.query(state, group["terrain_id"])
                state, logits = model.predict(
                    state, group["a"][:, step], normalize_xi(group["xi"][:, step]),
                    ground, support,
                )
                target = unflatten_state(group["s1"][:, step])
                errors.append(model.state_error(state, target).square().mean())
                joint_errors.append((state[4] - target[4]).square().mean())
                contact_errors.append(
                    ((logits > 0) != group["contacts1"][:, step]).float().mean()
                )
            values = {
                "windows": len(group["s0"]),
                "window_steps": group["a"].shape[1],
                "state_mse": torch.stack(errors).mean().item(),
                "last_state_mse": errors[-1].item(),
                "joint_rmse_rad": torch.stack(joint_errors).mean().sqrt().item(),
                "contact_accuracy": 1 - torch.stack(contact_errors).mean().item(),
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError(f"Nonfinite fixed world-model validation in {name}")
            result[name] = values
        return result
