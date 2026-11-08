"""Go2 tensor/learning regressions; no robot assets or simulator required."""

import io
import unittest
from dataclasses import replace
from itertools import pairwise

import torch

from supermimic.go2.buffer import FIELDS, ReplayBuffer
from supermimic.go2.commands import advance_commands, sample_commands
from supermimic.go2.config import OBS_DIM, CommandCfg, NetworkCfg
from supermimic.go2.learner import Learner
from supermimic.go2.mdp import (
    VelocityMDP,
    flatten_state,
    invariant_features,
    unflatten_state,
)
from supermimic.go2.networks import RunningNorm
from supermimic.go2.terrain import TILE_NAMES, mixed_profile, profiles
from supermimic.rotations import so3_exp


def standing(count=4):
    return (
        torch.tensor([[0.0, 0.0, 0.34]]).repeat(count, 1),
        torch.eye(3).repeat(count, 1, 1),
        torch.zeros(count, 3),
        torch.zeros(count, 3),
        torch.zeros(count, 12),
        torch.zeros(count, 12),
    )


class CourseProfileTest(unittest.TestCase):
    def test_default_course_keeps_its_section_order_and_length(self):
        _, sections = mixed_profile(profiles(), TILE_NAMES)
        self.assertEqual(tuple(name for name, _ in sections), TILE_NAMES)
        self.assertEqual(sections[-1][1], 23.96)

    def test_repeated_stairs_keep_vertical_risers_and_level_connections(self):
        sequence = ("stairs_down", "flat", "stairs_up_low", "stairs_down")
        points, sections = mixed_profile(profiles(), sequence)
        self.assertEqual(tuple(name for name, _ in sections), sequence)
        for (x0, z0), (x1, z1) in pairwise(points):
            self.assertGreaterEqual(x1, x0)
            self.assertNotEqual((x0, z0), (x1, z1))
        self.assertEqual(sum(x0 == x1 for (x0, _), (x1, _) in pairwise(points)), 18)
        self.assertAlmostEqual(points[-1][1], -0.72)


class Go2Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.mdp = VelocityMDP(torch.zeros(12), -torch.ones(12) * 2, torch.ones(12) * 2)

    def test_normalizer_resume_preserves_freeze_threshold(self):
        normalizer = RunningNorm(2)
        normalizer.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        saved = normalizer.state_dict()
        saved["count"] = torch.tensor(2e8)
        restored = RunningNorm(2)
        restored.load_state_dict(saved)
        restored.update(torch.full((3, 2), 100.0))
        torch.testing.assert_close(restored.mean, saved["mean"])
        torch.testing.assert_close(restored.var, saved["var"])
        torch.testing.assert_close(restored.count, saved["count"])

    def test_heading_invariance_and_velocity_objective(self):
        state = list(standing())
        state[2][:, 0] = 0.7
        action, xi = torch.zeros(4, 12), torch.zeros(4, 13)
        command = torch.tensor([[0.7, 0.0, 0.0]]).repeat(4, 1)
        yaw = so3_exp(torch.tensor([[0.0, 0.0, 1.2]])).expand(4, -1, -1)
        rotated = list(state)
        rotated[0] = (yaw @ state[0][..., None])[..., 0] + torch.tensor(
            [10.0, -7.0, 0.0]
        )
        rotated[1] = yaw @ state[1]
        rotated[2] = (yaw @ state[2][..., None])[..., 0]
        rotated[3] = (yaw @ state[3][..., None])[..., 0]
        torch.testing.assert_close(
            invariant_features(tuple(state)), invariant_features(tuple(rotated))
        )
        torch.testing.assert_close(
            self.mdp.obs(tuple(state), action, xi),
            self.mdp.obs(tuple(rotated), action, xi),
        )
        reward = self.mdp.reward(tuple(state), command, action, action)
        torch.testing.assert_close(
            reward, self.mdp.reward(tuple(rotated), command, action, action)
        )
        self.assertTrue(
            (reward > self.mdp.reward(tuple(state), -command, action, action)).all()
        )
        self.assertEqual(self.mdp.obs(tuple(state), action, xi).shape[-1], OBS_DIM)
        for actual, expected in zip(
            unflatten_state(flatten_state(tuple(state))), state
        ):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_command_boundary_preserves_unexpired_and_resamples_due(self):
        cfg = CommandCfg(
            vx=(0.6, 0.6),
            vy=(0.0, 0.0),
            yaw=(0.0, 0.0),
            min_steps=3,
            max_steps=3,
            standing_probability=0,
            curriculum_iters=10,
        )
        old = torch.tensor([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
        new, timer = advance_commands(old, torch.tensor([1, 2]), cfg, scale=0.5)
        torch.testing.assert_close(
            new, torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
        )
        self.assertEqual(timer.tolist(), [3, 1])
        self.assertAlmostEqual(cfg.scale(0), 0.2)
        self.assertAlmostEqual(cfg.scale(10), 1.0)
        command, _ = sample_commands(8, "cpu", replace(cfg, standing_probability=1))
        self.assertEqual(command.count_nonzero().item(), 0)

    def make_buffer(self):
        buffer = ReplayBuffer(7, 4, "cpu")
        for step in range(12):
            state = standing()
            following = list(standing())
            action = torch.randn(4, 12) * 0.1
            # Synthetic action-dependent dynamics exercise actual model fitting and BPTT.
            following[2] = action[:, :3] * 0.05
            following[4] = action * 0.01
            transition = {
                key: torch.zeros((4,) + (() if shape == () else (shape,)), dtype=dtype)
                for key, (shape, dtype) in FIELDS.items()
            }
            transition.update(
                s=flatten_state(state),
                s1=flatten_state(tuple(following)),
                a=action,
                command=torch.tensor([[0.3, 0.0, 0.0]]).repeat(4, 1),
                remaining=torch.full((4,), 1 + step % 3, dtype=torch.long),
                ep_step=torch.full((4,), step % 3, dtype=torch.long),
                done=torch.full((4,), step % 3 == 2, dtype=torch.bool),
            )
            transition["obs"] = (
                self.mdp.obs(state, transition["a_prev"], transition["xi"])
                + step * 0.01
            )
            buffer.add(transition)
        return buffer

    def test_wrapped_replay_windows_and_histories_respect_resets(self):
        buffer = self.make_buffer()
        windows = buffer.sample_windows(64, 4)
        done, mask = windows["done"], windows["mask"]
        for step in range(1, 4):
            torch.testing.assert_close(mask[:, step], (~done[:, :step].any(-1)).float())
        # Observation's first joint offset encodes chronological step / 100.
        _, history = buffer.sample_starts(256, 5)
        values = (history[:, :, 0] * 100).round().long()
        for row in values.tolist():
            episode_start = row[-1] - row[-1] % 3
            self.assertGreaterEqual(min(row), max(episode_start, 5))
            self.assertEqual(row, sorted(row))
            self.assertTrue(
                all(right - left in (0, 1) for left, right in pairwise(row))
            )

    def test_model_fit_policy_gradient_and_optimizer_resume(self):
        cfg = NetworkCfg(
            hist=3,
            latent=8,
            wm_hidden=(32, 32),
            encoder_hidden=(16,),
            decoder_hidden=(32, 16),
            critic_hidden=(16,),
        )
        learner = Learner(self.mdp, cfg, CommandCfg(min_steps=2, max_steps=3), "cpu")
        buffer = self.make_buffer()
        for row in range(buffer.size):
            learner.observe_transition(
                {key: value[row] for key, value in buffer.data.items()}
            )
        for _ in range(3):
            metrics = learner.wm_update(buffer, 16, 2)
            self.assertTrue(torch.isfinite(metrics["wm_loss"]))
        model_before = {
            key: value.clone() for key, value in learner.wm.state_dict().items()
        }
        encoder_before = [p.detach().clone() for p in learner.pi.enc.parameters()]
        decoder_before = [p.detach().clone() for p in learner.pi.dec.parameters()]
        metrics = learner.policy_update(buffer, 8, 4, critic_epochs=1, minibatches=2)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(encoder_before, learner.pi.enc.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(decoder_before, learner.pi.dec.parameters())
            )
        )
        for key, value in learner.wm.state_dict().items():
            torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(learner.checkpoint(), stream)
        stream.seek(0)
        restored = Learner(self.mdp, cfg, learner.commands, "cpu")
        restored.restore(torch.load(stream, weights_only=True))
        start, history = buffer.sample_starts(4, cfg.hist)
        torch.testing.assert_close(
            restored.pi(history, start["command"]),
            learner.pi(history, start["command"]),
        )
        self.assertTrue(restored.opt_wm.state)
        self.assertTrue(restored.opt_pi.state)
        self.assertTrue(restored.opt_v.state)

    def test_fall_termination_and_soft_survival(self):
        state = list(standing(3))
        state[0][1, 2] = 0.05
        state[1][2] = so3_exp(torch.tensor([1.6, 0.0, 0.0]))
        self.assertEqual(
            self.mdp.termination(tuple(state)).tolist(), [False, True, True]
        )
        probability = self.mdp.soft_termination(tuple(state))
        self.assertLess(probability[0], 0.001)
        self.assertTrue((probability[1:] > 0.99).all())


if __name__ == "__main__":
    unittest.main()
