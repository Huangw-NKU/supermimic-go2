"""Differentiable velocity rewards and observations shared by sim and imagination."""

import torch

from supermimic.rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat

from .config import CONTROL_DT, NUM_J, TaskCfg


def flatten_state(state):
    p, R, v, w, q, qd = state
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x):
    return (
        x[..., :3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13:25],
        x[..., 25:37],
    )


def invariant_features(state, support=0.0):
    p, R, v, w, q, qd = state
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            (p[..., 2] - support)[..., None],
            mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R),
            (Rt @ v[..., None])[..., 0],
            (Rt @ w[..., None])[..., 0],
            q,
            qd,
        ),
        -1,
    )


def normalize_xi(xi):
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:] / 0.01), -1)


ROUTE_DIM = 3


def route_features(state, heading):
    position, rotation = state[:2]
    front = rotation[..., :2, 0]
    front = front / front.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    side = torch.stack((-front[..., 1], front[..., 0]), dim=-1)
    normal = torch.stack((-heading[..., 1], heading[..., 0]), dim=-1)
    return torch.stack(
        (
            (front * heading).sum(-1),
            (side * heading).sum(-1),
            (position[..., :2] * normal).sum(-1),
        ),
        dim=-1,
    )


def route_reward(state, command, heading):
    """Reward desired-direction speed and penalize cross-track error on guided trials."""
    position, _, velocity, *_ = state
    along = (velocity[..., :2] * heading).sum(-1)
    normal = torch.stack((-heading[..., 1], heading[..., 0]), dim=-1)
    lateral = (position[..., :2] * normal).sum(-1)
    active = heading.square().sum(-1) > 0
    progress = 0.6 * torch.exp(-((along - command[..., 0]) / 0.35).square())
    drift = 0.75 * torch.nn.functional.smooth_l1_loss(
        lateral, torch.zeros_like(lateral), beta=0.4, reduction="none"
    )
    return CONTROL_DT * torch.where(active, progress - drift, 0.0)


class VelocityMDP:
    def __init__(self, default_q, lower, upper, cfg: TaskCfg | None = None):
        self.cfg = cfg or TaskCfg()
        self.default_q = default_q
        mid, half = (lower + upper) / 2, (upper - lower) / 2
        self.soft_lo, self.soft_hi = mid - 0.9 * half, mid + 0.9 * half
        self.obs_noise = default_q.new_tensor(
            [0.01] * NUM_J + [0.5] * NUM_J + [0.2] * 3 + [0.05] * 3 + [0.0] * NUM_J
        )

    def obs(self, state, a_prev, xi, contacts=None):
        _, R, _, w, q, qd = state
        fields = (
            q - self.default_q - xi[..., 1:],
            qd,
            (R.transpose(-1, -2) @ w[..., None])[..., 0],
            -R[..., 2, :],
            a_prev,
        )
        if contacts is not None:
            fields += (contacts,)
        return torch.cat(fields, -1)

    def add_noise(self, obs):
        proprio = obs[..., : self.obs_noise.numel()]
        noisy = proprio + (2 * torch.rand_like(proprio) - 1) * self.obs_noise
        return torch.cat((noisy, obs[..., self.obs_noise.numel() :]), -1)

    @staticmethod
    def velocity(state):
        _, R, v, w, _, _ = state
        heading_v = (yaw_mat(R).transpose(-1, -2) @ v[..., None])[..., 0]
        return torch.cat((heading_v[..., :2], w[..., 2:3]), -1)

    def reward(self, state, command, action, previous, *, terms=False, support=0.0):
        p, R, v, w, q, _ = state
        cfg = self.cfg
        error = self.velocity(state) - command
        linear_error = error[..., :2].square().sum(-1)
        yaw_error = error[..., 2].square()
        body_w = (R.transpose(-1, -2) @ w[..., None])[..., 0]
        posture = (q - self.default_q).square().sum(-1)
        standing = (command.square().sum(-1) < 1e-6).to(q.dtype)
        reward = (
            cfg.w_linear * torch.exp(-linear_error / cfg.linear_std**2)
            + cfg.w_angular * torch.exp(-yaw_error / cfg.angular_std**2)
            - cfg.w_vertical * v[..., 2].square()
            - cfg.w_roll_pitch_rate * body_w[..., :2].square().sum(-1)
            - cfg.w_orientation * R[..., 2, :2].square().sum(-1)
            - cfg.w_height * (p[..., 2] - support - cfg.target_height).square()
            - cfg.w_action_rate * (action - previous).square().sum(-1)
            - cfg.w_joint_limit
            * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
            - (cfg.w_posture + cfg.w_standing * standing) * posture
        ) * CONTROL_DT
        if terms:
            return reward, {
                "linear_error": linear_error.sqrt(),
                "yaw_error": yaw_error.sqrt(),
            }
        return reward

    def termination(self, state, support=0.0):
        return (state[0][..., 2] - support < self.cfg.min_height) | (
            state[1][..., 2, 2] < self.cfg.min_upright
        )

    def soft_termination(self, state, support=0.0):
        height_alive = torch.sigmoid(
            (state[0][..., 2] - support - self.cfg.min_height) / 0.015
        )
        tilt_alive = torch.sigmoid((state[1][..., 2, 2] - self.cfg.min_upright) / 0.04)
        return 1 - height_alive * tilt_alive

    def critic_features(
        self, state, a_prev, command, remaining, xi, max_command_steps, support=0.0
    ):
        # Previous action matters to the action-rate reward; the timer matters to command changes.
        return torch.cat(
            (
                invariant_features(state, support),
                a_prev,
                command,
                remaining[..., None].to(command.dtype) / max_command_steps,
                normalize_xi(xi),
            ),
            -1,
        )
