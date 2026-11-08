"""Simulator-independent model fitting and analytic policy-gradient updates."""

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .commands import advance_commands
from .config import NUM_FEET, NUM_J, OBS_DIM, CommandCfg, NetworkCfg, TaskCfg
from .mdp import ROUTE_DIM, normalize_xi, route_features, route_reward, unflatten_state
from .networks import Critic, VelocityPolicy, WorldModel
from .terrain import TERRAIN_DIM


def requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class Learner:
    def __init__(
        self,
        mdp,
        networks: NetworkCfg,
        commands: CommandCfg,
        device,
        *,
        lr=2e-4,
        gamma=0.99,
        lam=0.95,
        compile_model=False,
        terrain=None,
        route_guidance=False,
        retention=None,
        retention_weight=0.1,
        retention_batch=1024,
        retention_budget=None,
        real_critic_weight=0.0,
        real_critic_batch=512,
        real_critic_window=32,
    ):
        self.mdp, self.cfg, self.commands = mdp, networks, commands
        self.gamma, self.lam = gamma, lam
        self.terrain = terrain
        self.route_guidance = route_guidance
        self.retention = retention
        self.retention_weight = retention_weight
        self.retention_batch = retention_batch
        self.retention_budget = retention_budget
        self.real_critic_weight = real_critic_weight
        self.real_critic_batch = real_critic_batch
        self.real_critic_window = real_critic_window
        terrain_dim = TERRAIN_DIM if terrain is not None else 0
        self.wm = WorldModel(networks, terrain_dim).to(device)
        self.pi = VelocityPolicy(
            networks, mdp.cfg.action_clip, foot_contacts=terrain is not None
        ).to(device)
        self.critic = Critic(networks, terrain_dim + (ROUTE_DIM if route_guidance else 0)).to(device)
        scales = [
            max(abs(lo), abs(hi), 0.1)
            for lo, hi in (commands.vx, commands.vy, commands.yaw)
        ]
        self.pi.command_scale.copy_(torch.tensor(scales, device=device))
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.model_step = (
            torch.compile(self._model_step) if compile_model else self._model_step
        )
        self.actor_step = (
            torch.compile(self._actor_step) if compile_model else self._actor_step
        )
        self.reward_term = (
            torch.compile(self._reward_term) if compile_model else self._reward_term
        )
        self.error_floor = torch.tensor(
            [3e-3] * 3
            + [5e-3] * 3
            + [2e-2] * 3
            + [5e-2] * 3
            + [3e-3] * NUM_J
            + [5e-2] * NUM_J,
            device=device,
        )

    @torch.no_grad()
    def observe_transition(self, transition):
        state, following = (
            unflatten_state(transition["s"]),
            unflatten_state(transition["s1"]),
        )
        ground, support = self._ground(state, transition.get("terrain_id"))
        self.wm.in_norm.update(
            self.wm.inputs(
                state, transition["a"], normalize_xi(transition["xi"]), ground, support
            )
        )
        self.wm.out_norm.update(self.wm.targets(state, following))
        self.pi.obs_norm.update(transition["obs"])

    def _ground(self, state, tile):
        return (None, 0.0) if self.terrain is None else self.terrain.query(state, tile)

    def _model_step(self, state, action, xi, tile):
        ground, support = self._ground(state, tile)
        return self.wm.predict(state, action, xi, ground, support)

    def wm_update(self, buffer, batch, window):
        samples = buffer.sample_windows(batch, window)
        state = unflatten_state(samples["s"][:, 0])
        predictions = []
        contacts = []
        tile = samples.get("terrain_id")
        for step in range(window):
            state, logits = self.model_step(
                state, samples["a"][:, step], normalize_xi(samples["xi"][:, step]), tile
            )
            predictions.append(state)
            contacts.append(logits)
        predicted = tuple(
            torch.stack([state[i] for state in predictions], 1) for i in range(6)
        )
        error = self.wm.state_error(predicted, unflatten_state(samples["s1"]))
        error = (
            (error / torch.maximum(self.wm.out_norm.std(), self.error_floor))
            .square()
            .mean(-1)
        )
        mask = samples["mask"]
        count = mask.sum().clamp_min(1)
        loss = (error * mask).sum() / count
        metrics = {}
        if tile is not None:
            logits = torch.stack(contacts, 1)
            target = samples["contacts1"]
            contact_error = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            ).mean(-1)
            contact_loss = (contact_error * mask).sum() / count
            loss = loss + 0.25 * contact_loss
            accuracy = ((logits > 0) == target).float().mean(-1)
            metrics.update(
                wm_contact_loss=contact_loss.detach(),
                wm_contact_accuracy=(accuracy * mask).sum() / count,
            )
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(
                "World-model update produced nonfinite loss/gradients"
            )
        self.opt_wm.step()
        last = (error[:, -1] * mask[:, -1]).sum() / mask[:, -1].sum().clamp_min(1)
        return {"wm_loss": loss.detach(), "wm_last_error": last.detach(), **metrics}

    def critic_input(self, state, previous, command, remaining, xi, tile=None, heading=None):
        ground, support = self._ground(state, tile)
        features = self.mdp.critic_features(
            state, previous, command, remaining, xi, self.commands.max_steps, support
        )
        if ground is not None:
            features = torch.cat((features, ground), dim=-1)
        if heading is not None:
            features = torch.cat((features, route_features(state, heading)), dim=-1)
        return features

    def _actor_step(self, state, history, command, xi, xi_normalized, tile):
        action = self.pi(history, command)
        following, logits = self._model_step(state, action, xi_normalized, tile)
        contacts = None
        if tile is not None:
            probability = logits.sigmoid()
            # Match the binary sensor forward pass while propagating a smooth gradient.
            contacts = (logits > 0).to(logits.dtype) + (
                probability - probability.detach()
            )
        obs = self.mdp.add_noise(self.mdp.obs(following, action, xi, contacts))
        return following, action, torch.cat((history[:, 1:], obs[:, None]), 1)

    def _reward_term(self, state, command, action, previous, tile, heading=None):
        _, support = self._ground(state, tile)
        reward = self.mdp.reward(state, command, action, previous, support=support)
        if heading is not None:
            reward = reward + route_reward(state, command, heading)
        return reward, self.mdp.soft_termination(state, support)

    def policy_update(
        self,
        buffer,
        batch,
        horizon,
        *,
        command_scale=1.0,
        critic_epochs=2,
        minibatches=4,
        diagnostics=False,
    ):
        start, history = buffer.sample_starts(batch, self.cfg.hist)
        # AOTAutograd backward buffers are single-use. Periodic multi-gradient
        # diagnostics use the same eager expressions; ordinary updates compile.
        actor_step = self._actor_step if diagnostics else self.actor_step
        reward_term = self._reward_term if diagnostics else self.reward_term
        state = unflatten_state(start["s"])
        xi, previous, command, remaining = (
            start[k] for k in ("xi", "a_prev", "command", "remaining")
        )
        xi_n = normalize_xi(xi)
        tile = start.get("terrain_id")
        heading = start.get("route_heading")
        fixed = None if tile is None else tile != 0
        states, previous_actions = [state], [previous]
        commands, timers, actions = [command], [remaining], []
        requires_grad(self.wm, False)
        requires_grad(self.critic, False)
        try:
            for _ in range(horizon):
                state, action, history = actor_step(
                    state, history, command, xi, xi_n, tile
                )
                states.append(state)
                actions.append(action)
                previous_actions.append(action)
                previous = action
                command, remaining = advance_commands(
                    command, remaining, self.commands, command_scale, fixed
                )
                commands.append(command)
                timers.append(remaining)
            # Evaluate analytic heads in one batch instead of dispatching kernels
            # for every imagined step on the host.
            flat = lambda values: torch.stack(values).flatten(0, 1)
            following = tuple(flat([s[i] for s in states[1:]]) for i in range(6))
            reward, terminal = reward_term(
                following,
                flat(commands[:-1]),
                flat(actions),
                flat(previous_actions[:-1]),
                None if tile is None else tile.repeat(horizon),
                None if heading is None else heading.repeat(horizon, 1),
            )
            reward, terminal = (
                reward.view(horizon, batch),
                terminal.view(horizon, batch),
            )
            survival = torch.cumprod(1 - terminal, 0)
            before = torch.cat((torch.ones_like(survival[:1]), survival[:-1]), 0)
            discount = self.gamma ** torch.arange(
                horizon, device=reward.device, dtype=reward.dtype
            )
            objective = (discount[:, None] * before * reward).sum(0)
            final_features = self.critic_input(
                state, previous, command, remaining, xi, tile, heading
            )
            objective += (
                self.gamma**horizon * survival[-1] * self.critic(final_features)
            )
            loss = -objective.mean()
            rl_loss = loss
            retention_metrics = {}
            if self.retention is not None:
                retained_history, retained_command, target = self.retention.sample(
                    self.retention_batch
                )
                retained_action = self.pi(retained_history, retained_command)
                retention_loss = (retained_action - target).square().mean()
                weight = self.retention_weight
                if self.retention_budget is not None:
                    group_mse = self.retention.group_mse(retained_action, target)
                    ratio = group_mse.detach().max().item() / self.retention_budget**2
                    weight = min(20.0, max(0.1, self.retention_weight * (ratio / 0.25)**2))
                    retention_metrics.update(
                        retention_weight=retention_loss.new_tensor(weight),
                        retention_worst_rmse=group_mse.detach().max().sqrt(),
                    )
                    if diagnostics:
                        parameters = list(self.pi.parameters())
                        main = torch.autograd.grad(rl_loss, parameters, retain_graph=True)
                        anchor = torch.autograd.grad(retention_loss, parameters, retain_graph=True)
                        main_norm = torch.stack([g.square().sum() for g in main]).sum().sqrt()
                        anchor_norm = torch.stack([g.square().sum() for g in anchor]).sum().sqrt()
                        retention_metrics["retention_grad_ratio"] = (weight * anchor_norm / main_norm.clamp_min(1e-12)).detach()
                loss = loss + weight * retention_loss
                retention_metrics.update({
                    "retention_loss": retention_loss.detach(),
                    "retention_rmse": retention_loss.detach().sqrt(),
                })
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(
                    "Policy update produced nonfinite loss/gradients"
                )
            if self.retention_budget is not None:
                from .policy_guard import budgeted_step
                guarded, scale = budgeted_step(
                    self.pi, self.opt_pi, lambda: self.retention.guard_mse(self.pi), self.retention_budget,
                )
                retention_metrics.update(
                    retention_guard_rmse=guarded.max().sqrt(),
                    pi_step_scale=loss.new_tensor(scale),
                    pi_step_rejected=loss.new_tensor(scale == 0.0),
                )
            else:
                self.opt_pi.step()
        finally:
            requires_grad(self.wm, True)
            requires_grad(self.critic, True)

        with torch.no_grad():
            all_states = tuple(flat([s[i].detach() for s in states]) for i in range(6))
            all_xi = xi.unsqueeze(0).expand(horizon + 1, -1, -1).flatten(0, 1)
            inputs = self.critic_input(
                all_states,
                flat(previous_actions).detach(),
                flat(commands),
                flat(timers),
                all_xi,
                None if tile is None else tile.repeat(horizon + 1),
                None if heading is None else heading.repeat(horizon + 1, 1),
            ).view(horizon + 1, batch, -1)
        self.critic.norm.update(inputs)
        with torch.no_grad():
            value = self.critic(inputs)
            target = value[-1]
            targets = []
            for step in range(horizon - 1, -1, -1):
                target = reward[step].detach() + self.gamma * (
                    1 - terminal[step].detach()
                ) * ((1 - self.lam) * value[step + 1] + self.lam * target)
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten()
        inputs = inputs[:-1].flatten(0, 1)
        real_inputs, real_targets, real_terminal = None, None, None
        if self.real_critic_weight:
            from .real_value import real_value_targets
            strata = ()
            if self.terrain is not None and hasattr(self.terrain, "family"):
                tiles = torch.arange(len(self.terrain.family), device=inputs.device) + self.terrain.first_tile
                strata = tuple(tiles[(self.terrain.family == family) & (self.terrain.level == 5)] for family in (0, 1))
            samples = buffer.sample_real_windows(self.real_critic_batch, self.real_critic_window, strata)
            real_inputs, real_targets, real_terminal = real_value_targets(self, samples)
            self.critic.norm.update(real_inputs)
        value_loss = reward.new_zeros(())
        real_loss = reward.new_zeros(())
        real_bias = reward.new_zeros(())
        for _ in range(critic_epochs):
            permutation = torch.randperm(len(inputs), device=inputs.device)
            chunks = permutation.tensor_split(min(minibatches, len(inputs)))
            real_chunks = None if real_inputs is None else torch.randperm(len(real_inputs), device=inputs.device).tensor_split(len(chunks))
            for chunk, indices in enumerate(chunks):
                value_loss = (
                    (self.critic(inputs[indices]) - targets[indices]).square().mean()
                )
                combined_value_loss = value_loss
                if real_chunks is not None:
                    ri = real_chunks[chunk]
                    real_error = self.critic(real_inputs[ri]) - real_targets[ri]
                    real_loss, real_bias = real_error.square().mean(), real_error.mean()
                    combined_value_loss = value_loss + self.real_critic_weight * real_loss
                self.opt_v.zero_grad(set_to_none=True)
                combined_value_loss.backward()
                value_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if not torch.isfinite(combined_value_loss) or not torch.isfinite(value_norm):
                    raise FloatingPointError(
                        "Critic update produced nonfinite loss/gradients"
                    )
                self.opt_v.step()
        real_metrics = {} if real_inputs is None else {
            "real_v_loss": real_loss.detach(), "real_v_bias": real_bias.detach(),
            "real_v_terminal_fraction": real_terminal.mean().detach(),
        }
        return {
            "pi_loss": loss.detach(),
            "v_loss": value_loss.detach(),
            "pi_gradnorm": norm.detach(),
            "imag_reward": reward.detach().mean(),
            "imag_survival": survival[-1].detach().mean(),
            **retention_metrics,
            **real_metrics,
        }

    def checkpoint(self):
        return {
            "format": "supermimic-go2-v1",
            "network_cfg": asdict(self.cfg),
            "task_cfg": asdict(self.mdp.cfg),
            "command_cfg": asdict(self.commands),
            "wm": self.wm.state_dict(),
            "pi": self.pi.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "opt_v": self.opt_v.state_dict(),
        }

    @torch.no_grad()
    def initialize_from_flat(self, checkpoint):
        """Transfer flat weights and normalizers into the expanded terrain networks."""
        if checkpoint.get("format") != "supermimic-go2-v1" or self.terrain is None:
            raise ValueError("Terrain initialization requires a Go2 flat checkpoint")
        source_task = TaskCfg(**checkpoint["task_cfg"])
        if source_task.terrain_course or checkpoint["network_cfg"] != asdict(self.cfg):
            raise ValueError(
                "Flat checkpoint must use the same network widths and history"
            )
        if (source_task.action_scale, source_task.action_clip) != (
            self.mdp.cfg.action_scale,
            self.mdp.cfg.action_clip,
        ):
            raise ValueError("Flat checkpoint must use the same action scale and clip")

        for name in ("wm", "pi", "critic"):
            module = getattr(self, name)
            source = checkpoint[name]
            destination = module.state_dict()
            for key, tensor in destination.items():
                old = source[key]
                if name == "pi" and key in ("enc.0.weight", "dec.0.weight"):
                    prefix = 3 if key.startswith("enc") else self.cfg.latent
                    tensor[:, :prefix].copy_(old[:, :prefix])
                    columns = tensor[:, prefix:].view(
                        tensor.shape[0], self.cfg.hist, OBS_DIM + NUM_FEET
                    )
                    columns.zero_()
                    columns[..., :OBS_DIM].copy_(
                        old[:, prefix:].reshape(tensor.shape[0], self.cfg.hist, OBS_DIM)
                    )
                elif tensor.shape == old.shape:
                    tensor.copy_(old)
                    # Keep the actor's pretrained normalization count. The fitted
                    # models need a short prior to adapt to the terrain data.
                    if name != "pi" and key.endswith(".count"):
                        tensor.clamp_(max=8192)
                else:
                    # Extra input weights and the new contact logits start at zero.
                    # Preserve unit variance for added normalizer channels.
                    tensor.fill_(1.0 if key.endswith(".var") else 0.0)
                    tensor[tuple(slice(0, dim) for dim in old.shape)].copy_(old)
            module.load_state_dict(destination)

    def restore(self, checkpoint):
        if checkpoint.get("format") != "supermimic-go2-v1":
            raise ValueError(
                "Expected a Go2 checkpoint, not a humanoid motion-tracking checkpoint"
            )
        for name in ("wm", "pi"):
            getattr(self, name).load_state_dict(checkpoint[name])
        critic = checkpoint["critic"]
        old_shape = critic["net.0.weight"].shape
        expand = self.route_guidance and old_shape[-1] + ROUTE_DIM == self.critic.norm.mean.numel()
        if expand:
            critic = dict(critic)
            critic["net.0.weight"] = F.pad(critic["net.0.weight"], (0, ROUTE_DIM))
            critic["norm.mean"] = F.pad(critic["norm.mean"], (0, ROUTE_DIM))
            critic["norm.var"] = F.pad(critic["norm.var"], (0, ROUTE_DIM), value=1.0)
        self.critic.load_state_dict(critic)
        for name in ("opt_wm", "opt_pi", "opt_v"):
            getattr(self, name).load_state_dict(checkpoint[name])
        if expand:
            first_layer = self.opt_v.state.get(self.critic.net[0].weight, {})
            for name, value in first_layer.items():
                if torch.is_tensor(value) and value.shape == old_shape:
                    first_layer[name] = F.pad(value, (0, ROUTE_DIM))
