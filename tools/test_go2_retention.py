"""Retention gradients and checkpoint compatibility without simulator assets."""

import tempfile
import unittest
from pathlib import Path

import torch

from supermimic.go2.buffer import ReplayBuffer
from supermimic.go2.config import CommandCfg, NetworkCfg, TaskCfg
from supermimic.go2.learner import Learner
from supermimic.go2.mdp import VelocityMDP, flatten_state
from supermimic.go2.retention import RETENTION_GROUPS, VALIDATION_GROUPS, RetentionData
from supermimic.go2.rough_curriculum import RoughCurriculum
from supermimic.go2.terrain import TerrainCourse
from tools.test_go2 import standing


class RetentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cfg = NetworkCfg(
            hist=3, latent=8, wm_hidden=(16,), encoder_hidden=(16,),
            decoder_hidden=(16,), critic_hidden=(16,),
        )
        self.mdp = VelocityMDP(
            torch.zeros(12), -torch.ones(12) * 2, torch.ones(12) * 2,
            TaskCfg(terrain_course=True),
        )
        self.learner = Learner(
            self.mdp, self.cfg, CommandCfg(), "cpu", terrain=TerrainCourse("cpu"),
        )
        self.checkpoint = self.learner.checkpoint()
        self.checkpoint["joint_names"] = [f"joint_{i}" for i in range(12)]
        policy = {}
        for index, name in enumerate(RETENTION_GROUPS):
            history = torch.zeros(4, 3, 46)
            history[:, :, 0] = index
            command = torch.tensor([[0.5, 0.0, 0.0]]).expand(4, -1).clone()
            with torch.no_grad():
                action = self.learner.pi(history, command)
            policy[name] = dict(history=history, command=command, action=action)
        state = flatten_state(standing())
        validation = {
            name: dict(
                s0=state, s1=state[:, None].expand(-1, 8, -1).clone(),
                a=torch.zeros(4, 8, 12), xi=torch.zeros(4, 8, 13),
                contacts1=torch.zeros(4, 8, 4, dtype=torch.bool),
                terrain_id=torch.zeros(4, dtype=torch.long),
            )
            for name in VALIDATION_GROUPS
        }
        self.path = Path(self.temporary.name) / "retention.pt"
        torch.save(dict(
            format="supermimic-go2-retention-v1", teacher_sha256="test",
            joint_names=self.checkpoint["joint_names"],
            network_cfg=self.checkpoint["network_cfg"],
            task_cfg=self.checkpoint["task_cfg"], policy=policy, validation=validation,
        ), self.path)

    def test_balanced_replay_and_incompatible_teacher(self):
        bank = RetentionData(self.path, "cpu", self.checkpoint)
        history, command, target = bank.sample(10)
        self.assertEqual(torch.bincount(history[:, 0, 0].long()).tolist(), [3, 3, 2, 2])
        torch.testing.assert_close(self.learner.pi(history, command), target)
        incompatible = dict(self.checkpoint, joint_names=list(reversed(self.checkpoint["joint_names"])))
        with self.assertRaisesRegex(ValueError, "joint_names"):
            RetentionData(self.path, "cpu", incompatible)

    def test_retention_reaches_the_normal_policy_update(self):
        bank = RetentionData(self.path, "cpu", self.checkpoint)
        self.learner.retention = bank
        self.learner.retention_batch = 8
        with torch.no_grad():
            self.learner.pi.dec[-1].bias.add_(0.1)
        buffer = ReplayBuffer(8, 4, "cpu", terrain_course=True)
        for tick in range(8):
            state = standing()
            action = torch.zeros(4, 12)
            transition = {
                key: torch.zeros_like(value[0]) for key, value in buffer.data.items()
            }
            transition.update(
                s=flatten_state(state), s1=flatten_state(state), a=action,
                obs=self.mdp.obs(state, action, torch.zeros(4, 13), torch.zeros(4, 4)),
                command=torch.tensor([[0.5, 0.0, 0.0]]).expand(4, -1),
                remaining=torch.full((4,), 100, dtype=torch.long),
                ep_step=torch.full((4,), tick, dtype=torch.long),
            )
            buffer.add(transition)
        targets = [group["action"].clone() for group in bank.groups]
        metrics = self.learner.policy_update(
            buffer, 4, 2, critic_epochs=1, minibatches=1,
        )
        self.assertGreater(metrics["retention_loss"].item(), 0)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        for group, target in zip(bank.groups, targets):
            torch.testing.assert_close(group["action"], target)
        report = bank.validate(self.learner.wm, self.learner.terrain)
        self.assertEqual(set(report), set(VALIDATION_GROUPS))
        self.assertTrue(all(row["window_steps"] == 8 for row in report.values()))

    def test_coverage_keeps_the_original_terrain_sampling(self):
        curriculum = RoughCurriculum.__new__(RoughCurriculum)
        TerrainCourse.__init__(curriculum, "cpu")
        curriculum.start, curriculum.ramp, curriculum.route_start = 20000, 3000, 32000
        curriculum.l5_focus_start = None
        curriculum.family = torch.tensor([0] * 4 + [1] * 4 + [2] * 8)
        curriculum.level = torch.tensor([2, 3, 4, 5] * 2 + [2, 3] * 4)
        curriculum.l5_coverage_start = None
        torch.manual_seed(47)
        original = curriculum.sample(30000, 55000, 4000, 9000)
        curriculum.l5_coverage_start = 51000
        torch.manual_seed(47)
        covered = curriculum.sample(30000, 55000, 4000, 9000)
        torch.testing.assert_close(original, covered)
        self.assertEqual(curriculum.l5_coverage(51000), 0.0)
        self.assertEqual(curriculum.l5_coverage(52000), 0.5)
        self.assertEqual(curriculum.l5_coverage(55000), 1.0)


if __name__ == "__main__":
    unittest.main()
