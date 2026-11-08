"""Run a terrain-trained Go2 checkpoint across Isaac Lab-style rough tiles."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from .config import CONTROL_DT, CommandCfg, NetworkCfg, TaskCfg
from .mdp import unflatten_state
from .networks import VelocityPolicy
from .rough import (
    FAMILIES,
    GOAL_EDGE,
    HEADINGS,
    LEVELS,
    MAX_CROSS_TRACK,
    OUTSIDE_EDGE,
    RoughCourse,
)

RESULTS = ("active", "passed", "fall", "off_course", "timeout")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--terrains", nargs="+", choices=FAMILIES, default=tuple(FAMILIES))
    parser.add_argument("--levels", nargs="+", type=int, choices=LEVELS, default=LEVELS)
    parser.add_argument(
        "--headings", nargs="+", type=int, default=HEADINGS, help="initial yaw angles in degrees"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=(42, 43, 44), help="seeds for random tiles"
    )
    parser.add_argument(
        "--speed", type=float, default=0.5, help="body-forward command in m/s"
    )
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="per-trial time limit"
    )
    parser.add_argument("--asset", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--real_time", action="store_true", help="pace GUI playback at 50 Hz")

    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not math.isfinite(args.seconds) or args.seconds < CONTROL_DT:
        parser.error("--seconds must be finite and at least 0.02 s")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("--seeds must be integers in [0, 2**32)")
    terrains = tuple(dict.fromkeys(args.terrains))
    levels = tuple(dict.fromkeys(args.levels))
    headings = tuple(dict.fromkeys(deg % 360 for deg in args.headings))
    seeds = tuple(dict.fromkeys(args.seeds))

    torch.set_num_threads(1)
    device = args.device or "cuda:0"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("Go2 Rough evaluation requires a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "supermimic-go2-v1":
        parser.error("Expected a Go2 checkpoint")
    task = TaskCfg(**checkpoint["task_cfg"])
    if not task.terrain_course:
        parser.error("Expected a terrain-trained Go2 checkpoint with four contact observations")
    network = NetworkCfg(**checkpoint["network_cfg"])
    command = CommandCfg(**checkpoint["command_cfg"])
    policy = VelocityPolicy(network, task.action_clip, foot_contacts=True).to(device).eval()
    policy.load_state_dict(checkpoint["pi"])

    from .env import VelocitySim, make_sim

    cfg = make_sim(device, terrain_course=True)
    cfg.render_interval = round(CONTROL_DT / cfg.dt)
    steps = round(args.seconds / CONTROL_DT)
    course = RoughCourse(device, terrains, dict.fromkeys(terrains, levels), headings, seeds)
    print(
        f"Go2 Rough: {len(course.cases)} trials, {len(course.origins)} tiles at {args.speed:g} m/s.",
        flush=True,
    )
    with launch_simulation(cfg, args), torch.no_grad():
        env = VelocitySim(
            cfg,
            len(course.cases),
            task,
            command,
            eval_mode=True,
            asset=args.asset or checkpoint["args"].get("asset"),
            rough_course=course,
        )
        if env.joint_names != checkpoint["joint_names"]:
            raise ValueError("Go2 joint ordering differs from the checkpoint")
        env.command.zero_()
        env.command[:, 0] = args.speed
        env.remaining.fill_(steps + 1)
        cases = evaluate(env, policy, network.hist, steps, args.real_time)

    if cases is None:
        return
    report = {
        "checkpoint": str(args.ckpt.resolve()),
        "iteration": checkpoint["iteration"],
        "mode": "deterministic Newton physics, nominal friction, zero sensor noise",
        "speed_m_s": args.speed,
        "max_seconds": steps * CONTROL_DT,
        "heading_deg": headings,
        "random_seeds": seeds,
        "criteria": {
            "goal_edge_m": GOAL_EDGE,
            "max_cross_track_m": MAX_CROSS_TRACK,
            "out_of_bounds_m": OUTSIDE_EDGE,
            "min_height_m": task.min_height,
            "min_upright_cos": task.min_upright,
        },
        "cases": cases,
    }
    destination = args.output or args.ckpt.with_suffix(".rough_eval.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    for family in terrains:
        for level in sorted({case["level"] for case in cases if case["family"] == family}):
            subset = [
                case for case in cases if case["family"] == family and case["level"] == level
            ]
            totals = dict(Counter(case["result"] for case in subset))
            print(f"{family:12} L{level} {subset[0]['parameter']:24} {totals}", flush=True)
    print(f"Report: {destination.resolve()}", flush=True)


def evaluate(env, policy, history_length, steps, real_time):
    course = env.rough_course
    status = torch.zeros(env.N, dtype=torch.long, device=env.device)
    ended_at = torch.zeros_like(status)
    end_progress = torch.zeros(env.N, 2, device=env.device)
    max_along = torch.zeros(env.N, device=env.device)
    forward_error = torch.zeros_like(max_along)
    samples = torch.zeros_like(status)
    history = None
    for step in range(steps):
        if not env.sim.is_running():
            print("Viewer closed before evaluation completed.", flush=True)
            return None
        started = time.perf_counter()
        observation = env.observe()
        history = (
            observation[:, None].expand(-1, history_length, -1)
            if history is None
            else torch.cat((history[:, 1:], observation[:, None]), dim=1)
        )
        active = status == 0
        action = policy(history, env.command)
        action[~active] = 0
        transition, _ = env.step(action, observation)
        state = unflatten_state(transition["s1"])
        if not torch.isfinite(transition["s1"][active]).all():
            raise FloatingPointError("Go2 Rough produced nonfinite physical state")
        along, lateral = course.progress(state[0])
        max_along[active] = torch.maximum(max_along[active], along[active])
        status[active & transition["term"]] = 2
        status[active & (status == 0) & transition["course_reached"]] = 1
        status[active & (status == 0) & transition["course_outside"]] = 3
        if step + 1 == steps:
            status[active & (status == 0)] = 4
        ended = active & (status != 0)
        ended_at[ended] = step + 1
        end_progress[ended] = torch.stack((along[ended], lateral[ended]), dim=1)
        measure = active & (status != 2)
        if step >= round(1 / CONTROL_DT):
            error = (env.mdp.velocity(state)[:, 0] - env.command[:, 0]).abs()
            forward_error[measure] += error[measure]
            samples[measure] += 1
        if env.sim.is_rendering:
            if env.N == 1:
                focus = (state[0][0] + env.origins[0]).tolist()
                hx, hy = course.heading[0].tolist()
                env.sim.set_camera_view(
                    eye=[
                        focus[0] - 2 * hx + 2 * hy,
                        focus[1] - 2 * hy - 2 * hx,
                        focus[2] + 1.6,
                    ],
                    target=focus,
                )
            env.sim.render()
        if not (status == 0).any():
            break
        if real_time:
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - started)))

    return [
        dict(
            case,
            result=RESULTS[result],
            duration_s=round(tick * CONTROL_DT, 3),
            goal_m=round(goal, 3),
            progress_m=round(along, 3),
            cross_track_m=round(lateral, 3),
            max_progress_m=round(best, 3),
            mean_vx_error_m_s=round(error / count, 4) if count else None,
        )
        for case, result, tick, (along, lateral), goal, best, error, count in zip(
            course.cases,
            status.tolist(),
            ended_at.tolist(),
            end_progress.tolist(),
            course.goal.tolist(),
            max_along.tolist(),
            forward_error.tolist(),
            samples.tolist(),
        )
    ]


if __name__ == "__main__":
    main()
