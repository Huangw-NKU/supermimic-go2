"""Evaluate deterministic Go2 velocity step responses in the actual simulator."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from .config import CONTROL_DT, CommandCfg, NetworkCfg, TaskCfg
from .mdp import unflatten_state
from .networks import VelocityPolicy
from .terrain import EVAL_TILE_NAMES, add_terrain_args, resolve_terrain_tile

DEFAULT_COMMANDS = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 0.75),
    (0.0, 0.0, -0.75),
    (0.5, 0.0, 0.5),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        help="one vx, vy, yaw command instead of the standard suite",
    )
    parser.add_argument(
        "--trials", type=int, default=1, help="deterministic repeats per command"
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--measure_seconds",
        type=float,
        default=2.0,
        help="steady-state window at the end of each trial",
    )
    parser.add_argument("--asset", default=None)
    add_terrain_args(parser)
    parser.add_argument("--output", default=None)
    parser.add_argument("--record", default=None, help="optional trajectory npz")
    parser.add_argument(
        "--real_time",
        action="store_true",
        help="pace evaluation at the 50 Hz control rate",
    )
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.terrain_tile = resolve_terrain_tile(parser, args)
    if (
        args.trials < 1
        or args.seconds < CONTROL_DT
        or not CONTROL_DT <= args.measure_seconds <= args.seconds
    ):
        parser.error("Require positive trials and 0.02 <= measure_seconds <= seconds")
    device = args.device or "cuda:0"
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    if checkpoint.get("format") != "supermimic-go2-v1":
        parser.error("Expected a Go2 checkpoint")
    network_cfg = NetworkCfg(**checkpoint["network_cfg"])
    task_cfg = TaskCfg(**checkpoint["task_cfg"])
    if args.terrain_tile is not None and not task_cfg.terrain_course:
        parser.error("--terrain_tile requires a Go2 terrain checkpoint")
    commands = torch.tensor(
        [args.command] if args.command is not None else DEFAULT_COMMANDS, device=device
    )
    per_env_commands = commands.repeat_interleave(args.trials, dim=0)
    steps = round(args.seconds / CONTROL_DT)
    measure_start = steps - round(args.measure_seconds / CONTROL_DT)
    command_cfg = replace(
        CommandCfg(**checkpoint["command_cfg"]),
        min_steps=steps + 1,
        max_steps=steps + 1,
    )
    policy = VelocityPolicy(
        network_cfg, task_cfg.action_clip, foot_contacts=task_cfg.terrain_course
    ).to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    torch.manual_seed(0)
    from isaaclab.app import launch_simulation

    from .env import VelocitySim, make_sim

    cfg = make_sim(device, terrain_course=task_cfg.terrain_course)
    cfg.render_interval = round(CONTROL_DT / cfg.dt)
    with launch_simulation(cfg, args), torch.no_grad():
        env = VelocitySim(
            cfg,
            len(per_env_commands),
            task_cfg,
            command_cfg,
            eval_mode=True,
            asset=args.asset or checkpoint["args"].get("asset"),
            terrain_tile=args.terrain_tile or "step_up",
            terrain_sequence=args.terrain_sequence,
        )
        if env.joint_names != checkpoint["joint_names"]:
            raise ValueError("Go2 joint ordering differs from the checkpoint")
        env.command.copy_(per_env_commands)
        env.remaining.fill_(steps + 1)
        history = None
        failed = torch.zeros(env.N, dtype=torch.bool, device=device)
        reached = torch.zeros_like(failed)
        velocity_sum = torch.zeros(env.N, 3, device=device)
        counts = torch.zeros(env.N, device=device)
        records = {"state": [], "alive": []}
        for step in range(steps):
            if not env.sim.is_running():
                print(
                    "Viewer closed; evaluation stopped before completion.", flush=True
                )
                return
            step_started = time.perf_counter()
            obs = env.observe()
            history = (
                obs[:, None].repeat(1, network_cfg.hist, 1)
                if history is None
                else torch.cat((history[:, 1:], obs[:, None]), 1)
            )
            action = policy(history, per_env_commands)
            action = torch.where(failed[:, None], torch.zeros_like(action), action)
            transition, _ = env.step(action, obs)
            if env.sim.is_rendering:
                if env.N == 1:
                    target = (transition["s1"][0, :3] + env.origins[0]).cpu().tolist()
                    eye = [target[0] + 2.0, target[1] - 2.0, target[2] + 1.2]
                    env.sim.set_camera_view(eye=eye, target=target)
                env.sim.render()
            failed |= transition["term"]
            if task_cfg.terrain_course:
                reached |= transition["course_reached"] & ~failed
            if step >= measure_start:
                velocity_sum += (
                    env.mdp.velocity(unflatten_state(transition["s1"]))
                    * (~failed)[:, None]
                )
                counts += ~failed
            if args.record:
                records["state"].append(transition["s1"].cpu())
                records["alive"].append((~failed).cpu())
            if args.real_time:
                time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - step_started)))
        mean_error = velocity_sum / counts.clamp_min(1)[:, None] - per_env_commands
        results = []
        for index, command in enumerate(commands):
            selection = slice(index * args.trials, (index + 1) * args.trials)
            survived = ~failed[selection]
            error = mean_error[selection][survived]
            results.append(
                {
                    "command": command.tolist(),
                    "survival_rate": float(survived.float().mean()),
                    "steady_linear_error_m_s": float(error[:, :2].norm(dim=-1).mean())
                    if len(error)
                    else None,
                    "steady_yaw_error_rad_s": float(error[:, 2].abs().mean())
                    if len(error)
                    else None,
                }
            )
            if task_cfg.terrain_course and env.eval_tile != 0:
                results[-1]["course_finish_rate"] = float(
                    reached[selection].float().mean()
                )
        report = {
            "checkpoint": str(Path(args.ckpt).resolve()),
            "iteration": checkpoint["iteration"],
            "seconds": steps * CONTROL_DT,
            "measure_seconds": args.measure_seconds,
            "trials": args.trials,
            "evaluation": "deterministic physics, no observation/action noise or domain randomization",
            "per_command": results,
        }
        if task_cfg.terrain_course:
            report["terrain_tile"] = EVAL_TILE_NAMES[env.eval_tile]
            if env.terrain.sections:
                report["terrain_sections"] = [
                    {"terrain": name, "end_x_m": end} for name, end in env.terrain.sections
                ]
        destination = (
            Path(args.output)
            if args.output
            else Path(args.ckpt).with_suffix(".velocity_eval.json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.record:
            import numpy as np

            record_path = Path(args.record)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                record_path,
                **{key: torch.stack(value).numpy() for key, value in records.items()},
                command=per_env_commands.cpu().numpy(),
                joint_names=env.joint_names,
                dt=CONTROL_DT,
            )
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
