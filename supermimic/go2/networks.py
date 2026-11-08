"""Go2 dynamics, velocity encoder/action decoder, and privileged critic."""

import torch
from torch import nn

from supermimic.rotations import so3_exp, so3_log, yaw_mat

from .config import INV_DIM, NUM_FEET, NUM_J, OBS_DIM, WM_OUT, XI_DIM, NetworkCfg
from .mdp import invariant_features


def mlp(inp, hidden, out, *, smooth=False, layer_norm=False):
    layers = []
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU() if smooth else nn.ELU())
        inp = width
    return nn.Sequential(*layers, nn.Linear(inp, out))


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self._samples_seen = 0.0

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._samples_seen = float(self.count)

    @torch.no_grad()
    def update(self, x):
        # A host counter avoids synchronizing CUDA on every normalization update.
        if self._samples_seen >= 2e8:
            return
        x = x.detach().reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        self._samples_seen += n
        total = self.count + n
        delta = x.mean(0) - self.mean
        self.var.copy_(
            (
                self.var * self.count
                + x.var(0, unbiased=False) * n
                + delta.square() * self.count * n / total
            )
            / total
        )
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x):
        return (x - self.mean) / self.std()


class WorldModel(nn.Module):
    def __init__(self, cfg: NetworkCfg, terrain_dim=0):
        super().__init__()
        input_dim = INV_DIM + NUM_J + XI_DIM + terrain_dim
        self.net = mlp(
            input_dim,
            cfg.wm_hidden,
            WM_OUT + (NUM_FEET if terrain_dim else 0),
            smooth=True,
            layer_norm=True,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(input_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    @staticmethod
    def inputs(state, action, xi_normalized, ground=None, support=0.0):
        fields = (invariant_features(state, support), action, xi_normalized)
        return torch.cat(fields if ground is None else (*fields, ground), -1)

    @staticmethod
    def targets(state, following):
        p, R, v, w, q, qd = state
        p1, R1, v1, w1, q1, qd1 = following
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (yaw_mat(R).transpose(-1, -2) @ (p1 - p)[..., None])[..., 0],
                so3_log(Rt @ R1),
                (Rt @ (v1 - v)[..., None])[..., 0],
                (Rt @ (w1 - w)[..., None])[..., 0],
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    @staticmethod
    def state_error(prediction, truth):
        return WorldModel.targets(truth, prediction)

    def step(self, state, action, xi_normalized):
        return self.predict(state, action, xi_normalized)[0]

    def predict(self, state, action, xi_normalized, ground=None, support=0.0):
        p, R, v, w, q, qd = state
        output = self.net(
            self.in_norm(self.inputs(state, action, xi_normalized, ground, support))
        )
        delta = output[..., :WM_OUT]
        delta = delta * self.out_norm.std() + self.out_norm.mean
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        following = (
            p + (yaw_mat(R) @ dp[..., None])[..., 0],
            R @ so3_exp(dr),
            v + (R @ dv[..., None])[..., 0],
            w + (R @ dw[..., None])[..., 0],
            q + dq,
            qd + dqd,
        )
        return following, output[..., WM_OUT:]


class VelocityPolicy(nn.Module):
    def __init__(self, cfg: NetworkCfg, action_clip=3.0, *, foot_contacts=False):
        super().__init__()
        self.hist, self.action_clip = cfg.hist, action_clip
        obs_dim = OBS_DIM + (NUM_FEET if foot_contacts else 0)
        self.obs_norm = RunningNorm(obs_dim)
        self.register_buffer("command_scale", torch.tensor([1.0, 1.0, 1.0]))
        self.enc = mlp(3 + obs_dim * cfg.hist, cfg.encoder_hidden, cfg.latent)
        self.dec = mlp(cfg.latent + obs_dim * cfg.hist, cfg.decoder_hidden, NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()

    def forward(self, history, command):
        obs = self.obs_norm(history).flatten(-2)
        latent = torch.tanh(
            self.enc(torch.cat((command / self.command_scale, obs), -1))
        )
        return self.action_clip * torch.tanh(
            self.dec(torch.cat((latent, obs), -1)) / self.action_clip
        )


CRITIC_DIM = INV_DIM + NUM_J + 3 + 1 + XI_DIM


class Critic(nn.Module):
    def __init__(self, cfg: NetworkCfg, terrain_dim=0):
        super().__init__()
        self.norm = RunningNorm(CRITIC_DIM + terrain_dim)
        self.net = mlp(CRITIC_DIM + terrain_dim, cfg.critic_hidden, 1)

    def forward(self, features):
        return self.net(self.norm(features)).squeeze(-1)
