"""Action-budget rollback and real-physics terminal/timeout target regressions."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from supermimic.go2.buffer import ReplayBuffer
from supermimic.go2.config import CommandCfg, NetworkCfg, TaskCfg
from supermimic.go2.learner import Learner
from supermimic.go2.mdp import VelocityMDP, flatten_state
from supermimic.go2.policy_guard import budgeted_step
from supermimic.go2.real_value import real_value_targets
from supermimic.go2.retention import RETENTION_GROUPS_V2, RetentionData
from supermimic.go2.terrain import TerrainCourse
from tools.test_go2 import standing


class GuardedLearningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_rejected_adam_candidate_restores_momentum_and_step(self):
        policy = nn.Linear(1, 1, bias=False)
        policy.weight.data.zero_()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.1)
        policy.weight.grad = torch.zeros_like(policy.weight)
        optimizer.step()
        previous = copy.deepcopy(optimizer.state_dict())
        policy.weight.grad.fill_(-1)
        mse, scale = budgeted_step(policy, optimizer, lambda: policy.weight.square().mean()[None], 1e-6)
        self.assertEqual(scale, 0)
        self.assertEqual(policy.weight.item(), 0)
        for key, value in previous['state'][0].items():
            torch.testing.assert_close(optimizer.state_dict()['state'][0][key], value, rtol=0, atol=0)
        self.assertEqual(mse.item(), 0)

    def test_backtracking_accepts_a_smaller_step_within_budget(self):
        policy = nn.Linear(1, 1, bias=False)
        policy.weight.data.zero_()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.1)
        policy.weight.grad = -torch.ones_like(policy.weight)
        mse, scale = budgeted_step(policy, optimizer, lambda: policy.weight.square().mean()[None], .02)
        self.assertGreater(scale, 0)
        self.assertLess(scale, 1)
        self.assertLessEqual(mse.sqrt().item(), .02 + 1e-7)
        self.assertEqual(optimizer.state[policy.weight]['step'].item(), 1)

    def test_real_target_stops_at_reset_but_bootstraps_a_timeout(self):
        samples = {
            's': flatten_state(standing(2))[:, None].repeat(1, 3, 1),
            's1': flatten_state(standing(2))[:, None].repeat(1, 3, 1),
            'a': torch.zeros(2, 3, 12), 'a_prev': torch.zeros(2, 3, 12),
            'xi': torch.zeros(2, 3, 13), 'command': torch.ones(2, 3, 3),
            'remaining': torch.ones(2, 3, dtype=torch.long),
            'command1': torch.tensor([[[3., 0, 0], [9., 0, 0], [99., 0, 0]]]).repeat(2, 1, 1),
            'remaining1': torch.ones(2, 3, dtype=torch.long),
            'r': torch.tensor([[1., 2, 1000], [1., 2, 1000]]),
            'done': torch.tensor([[False, True, False], [False, True, False]]),
            'term': torch.tensor([[False, True, False], [False, False, False]]),
            'terrain_id': torch.zeros(2, 3, dtype=torch.long),
            'route_heading': torch.zeros(2, 3, 2), 'mask': torch.tensor([[1., 1, 0], [1., 1, 0]]),
        }
        learner = SimpleNamespace(gamma=.5, critic=lambda features: features[:, 0],
                                  critic_input=lambda state, previous, command, *rest: command)
        _, targets, terminal = real_value_targets(learner, samples)
        torch.testing.assert_close(targets, torch.tensor([2., 4.25]))
        torch.testing.assert_close(terminal, torch.tensor([1., 0.]))

    def test_real_replay_wrap_mask_and_symmetric_l5_priority(self):
        buffer = ReplayBuffer(8, 6, 'cpu', terrain_course=True, route_guidance=True, real_targets=True)
        for tick in range(12):
            transition = {key: torch.zeros_like(value[0]) for key, value in buffer.data.items()}
            transition.update(terrain_id=torch.tensor([0, 11, 22, 0, 11, 22]), r=torch.arange(6).float()+tick*10,
                              done=torch.full((6,),tick%3==2), ep_step=torch.full((6,),tick%3))
            buffer.add(transition)
        windows = buffer.sample_real_windows(128, 4, (torch.tensor([11]), torch.tensor([22])))
        self.assertTrue((windows['terrain_id'][64:96, 0] == 11).all())
        self.assertTrue((windows['terrain_id'][96:, 0] == 22).all())
        torch.testing.assert_close(windows['r'][:, 1:] - windows['r'][:, :-1], torch.full((128, 3), 10.))
        for step in range(1,4):
            torch.testing.assert_close(windows['mask'][:, step], (~windows['done'][:, :step].any(-1)).float())
        before = {key:value.clone() for key,value in buffer.data.items()}
        pointer, size = buffer.ptr, buffer.size
        buffer.move_('cpu')
        self.assertEqual((buffer.ptr,buffer.size),(pointer,size))
        for key,value in before.items():
            torch.testing.assert_close(buffer.data[key],value,rtol=0,atol=0)

    def test_v2_bank_balances_up_down_and_actual_combined_update(self):
        cfg = NetworkCfg(hist=3, latent=8, wm_hidden=(16,), encoder_hidden=(16,), decoder_hidden=(16,), critic_hidden=(16,))
        mdp = VelocityMDP(torch.zeros(12), -torch.ones(12)*2, torch.ones(12)*2, TaskCfg(terrain_course=True))
        learner = Learner(mdp, cfg, CommandCfg(), 'cpu', terrain=TerrainCourse('cpu'), route_guidance=True,
                          retention_budget=.02, real_critic_weight=.25, real_critic_batch=8, real_critic_window=2)
        checkpoint = learner.checkpoint()
        checkpoint['joint_names'] = [str(i) for i in range(12)]
        history = torch.zeros(128, 3, 46)
        command = torch.tensor([[.5, 0, 0]]).repeat(128, 1)
        with torch.no_grad():
            labels = learner.pi(history, command)
        groups = {name: dict(history=history.clone(), command=command.clone(), action=labels.clone()) for name in RETENTION_GROUPS_V2}
        validation = {name: dict(s0=flatten_state(standing()), s1=flatten_state(standing())[:, None], a=torch.zeros(4, 1, 12), xi=torch.zeros(4, 1, 13), contacts1=torch.zeros(4, 1, 4, dtype=torch.bool), terrain_id=torch.zeros(4, dtype=torch.long)) for name in RETENTION_GROUPS_V2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bank.pt'
            torch.save(dict(format='supermimic-go2-retention-v2', teacher_sha256='test', joint_names=checkpoint['joint_names'], network_cfg=checkpoint['network_cfg'], task_cfg=checkpoint['task_cfg'], policy=groups, guard=groups, validation=validation), path)
            learner.retention = RetentionData(path, 'cpu', checkpoint)
            self.assertEqual(learner.retention.counts(1024), [256, 128, 256, 128, 128, 128])
            buffer = ReplayBuffer(8, 4, 'cpu', terrain_course=True, route_guidance=True, real_targets=True)
            for tick in range(8):
                transition = {key: torch.zeros_like(value[0]) for key, value in buffer.data.items()}
                transition.update(s=flatten_state(standing()), s1=flatten_state(standing()), obs=torch.zeros(4, 46),
                                  command=command[:4], command1=command[:4], remaining=torch.full((4,), 100), remaining1=torch.full((4,), 100), ep_step=torch.full((4,), tick), r=torch.full((4,), .02))
                buffer.add(transition)
            before = [p.detach().clone() for p in learner.critic.parameters()]
            metrics = learner.policy_update(buffer, 4, 2, critic_epochs=1, minibatches=2, diagnostics=True)
            self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
            self.assertIn('real_v_loss', metrics)
            self.assertIn('retention_grad_ratio', metrics)
            self.assertLessEqual(metrics['retention_guard_rmse'].item(), .020001)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, learner.critic.parameters(), strict=True)))


if __name__ == '__main__':
    unittest.main()
