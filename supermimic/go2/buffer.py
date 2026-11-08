"""GPU ring buffer; histories and model-fit windows never cross episode resets."""

import torch

from .config import NUM_FEET, NUM_J, OBS_DIM, STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (NUM_J, torch.float32),
    "a_prev": (NUM_J, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "obs": (OBS_DIM, torch.float32),
    "command": (3, torch.float32),
    "remaining": ((), torch.long),
    "ep_step": ((), torch.long),
    "r": ((), torch.float32),
    "done": ((), torch.bool),
    "term": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, steps, num_envs, device, *, terrain_course=False, route_guidance=False, real_targets=False):
        self.capacity, self.num_envs, self.device = steps, num_envs, device
        self.ptr = self.size = 0
        fields = FIELDS
        self.terrain_course = terrain_course
        self.route_guidance = route_guidance
        self.real_targets = real_targets
        if terrain_course:
            fields = {
                **FIELDS,
                "obs": (OBS_DIM + NUM_FEET, torch.float32),
                "terrain_id": ((), torch.long),
                "contacts1": (NUM_FEET, torch.bool),
            }
        if route_guidance:
            fields = {**fields, "route_heading": (2, torch.float32)}
        if real_targets:
            fields = {**fields, "command1": (3, torch.float32), "remaining1": ((), torch.long)}
        self.data = {
            key: torch.zeros(
                (steps, num_envs) + (() if shape == () else (shape,)),
                dtype=dtype,
                device=device,
            )
            for key, (shape, dtype) in fields.items()
        }

    def add(self, transition):
        for key, values in self.data.items():
            values[self.ptr].copy_(transition[key])
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _row(self, age):
        return (self.ptr - self.size + age) % self.capacity

    def move_(self, device):
        """Move storage intact while a separate evaluator needs GPU memory."""
        for key in self.data:
            self.data[key] = self.data[key].to(device)
        self.device = device
        return self

    def sample_windows(self, batch, window):
        if not 1 <= window <= self.size:
            raise ValueError("Replay must contain at least one full model-fit window")
        start = torch.randint(self.size - window + 1, (batch,), device=self.device)
        env = torch.randint(self.num_envs, (batch, 1), device=self.device)
        rows = self._row(start[:, None] + torch.arange(window, device=self.device))
        result = {
            key: self.data[key][rows, env] for key in ("s", "s1", "a", "xi", "done")
        }
        done = result["done"].float()
        if self.terrain_course:
            result["terrain_id"] = self.data["terrain_id"][rows[:, 0], env[:, 0]]
            result["contacts1"] = self.data["contacts1"][rows, env]
        result["mask"] = torch.cumprod(
            torch.cat((torch.ones_like(done[:, :1]), 1 - done[:, :-1]), -1), -1
        )
        return result

    def sample_starts(self, batch, hist):
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        age = torch.randint(self.size, (batch,), device=self.device)
        env = torch.randint(self.num_envs, (batch,), device=self.device)
        rows = self._row(age)
        back = torch.arange(hist - 1, -1, -1, device=self.device)[None, :]
        back = torch.minimum(back, self.data["ep_step"][rows, env, None])
        back = torch.minimum(back, age[:, None])
        history = self.data["obs"][self._row(age[:, None] - back), env[:, None]]
        start = {
            key: self.data[key][rows, env]
            for key in ("s", "a_prev", "xi", "command", "remaining")
        }
        if self.terrain_course:
            start["terrain_id"] = self.data["terrain_id"][rows, env]
        if self.route_guidance:
            start["route_heading"] = self.data["route_heading"][rows, env]
        return start, history

    def sample_real_windows(self, batch, window, strata=()):
        """True transitions; half uniform and equal priority strata when supplied."""
        if not self.real_targets or not 1 <= window <= self.size:
            raise ValueError("Real targets require stored next commands and full windows")
        span = self.size - window + 1
        sizes = [batch] if not strata else [batch // 2] + [
            (batch - batch // 2) // len(strata) + (i < (batch - batch // 2) % len(strata))
            for i in range(len(strata))
        ]
        ages, environments = [], []
        for index, count in enumerate(sizes):
            if index == 0:
                age = torch.randint(span, (count,), device=self.device)
                env = torch.randint(self.num_envs, (count,), device=self.device)
            else:
                proposed_age = torch.randint(span, (max(4096, count * 32),), device=self.device)
                proposed_env = torch.randint(self.num_envs, proposed_age.shape, device=self.device)
                tiles = self.data["terrain_id"][self._row(proposed_age), proposed_env]
                choices = torch.isin(tiles, strata[index - 1]).nonzero().flatten()
                if len(choices):
                    selected = choices[torch.randint(len(choices), (count,), device=self.device)]
                    age, env = proposed_age[selected], proposed_env[selected]
                else:
                    age = torch.randint(span, (count,), device=self.device)
                    env = torch.randint(self.num_envs, (count,), device=self.device)
            ages.append(age)
            environments.append(env)
        age, env = torch.cat(ages), torch.cat(environments)
        rows = self._row(age[:, None] + torch.arange(window, device=self.device))
        keys = ("s", "s1", "a", "a_prev", "xi", "command", "remaining", "command1", "remaining1", "r", "done", "term", "terrain_id", "route_heading")
        result = {key: self.data[key][rows, env[:, None]] for key in keys}
        done = result["done"].float()
        result["mask"] = torch.cumprod(torch.cat((torch.ones_like(done[:, :1]), 1 - done[:, :-1]), -1), -1)
        return result
