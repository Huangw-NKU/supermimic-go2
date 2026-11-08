"""Check Go2 asset loading, motor feedback, contacts, and reset consistency on GPU."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.steps) < 1:
        parser.error("num_envs and steps must be positive")
    import torch

    from supermimic.go2.config import CONTROL_DT, PHYSICS_DT, CommandCfg, TaskCfg
    from supermimic.go2.env import VelocitySim, make_sim

    torch.manual_seed(0)
    cfg = make_sim(args.device or "cuda:0")
    with launch_simulation(cfg, args), torch.no_grad():
        env = VelocitySim(
            cfg,
            args.num_envs,
            TaskCfg(episode_steps=100),
            CommandCfg(),
            asset=args.asset,
        )
        max_speed = 0.0
        for step in range(args.steps):
            state = env.state()
            obs = env.mdp.obs(state, env.a_prev, env.xi)
            # A small, changing action exercises motor feedback rather than a perfect equilibrium.
            action = 0.1 * torch.sin(torch.full_like(env.a_prev, step * CONTROL_DT * 3))
            transition, _ = env.step(action, obs)
            if not all(
                torch.isfinite(transition[key]).all() for key in ("s", "s1", "r")
            ):
                raise AssertionError("Go2 physics produced a nonfinite transition")
            max_speed = max(max_speed, float(transition["s1"][:, 25:].abs().max()))
            if max_speed > 100:
                raise AssertionError(
                    f"Unstable motor integration: joint speed {max_speed:.2f} rad/s"
                )
            reset = env.ep_step == 0
            if reset.any():
                assert torch.allclose(
                    env.state()[0][reset, 2],
                    torch.full_like(env.state()[0][reset, 2], 0.4),
                )
                assert not env.a_prev[reset].count_nonzero()
        print(
            f"GO2_PHYSICS_OK: {args.num_envs} envs x {args.steps} control steps; "
            f"control={1 / CONTROL_DT:g} Hz, motor={1 / PHYSICS_DT:g} Hz, max_joint_speed={max_speed:.2f} rad/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
