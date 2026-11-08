"""Train Go2 velocity control: python -m supermimic.go2.train --help."""

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import CommandCfg, NetworkCfg, TaskCfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument(
        "--iterations",
        type=int,
        default=50_000,
        help="total iteration target, including before resume",
    )
    parser.add_argument("--buffer_steps", type=int, default=500)
    parser.add_argument("--utd", type=float, default=0.4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--wm_hidden", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--encoder_hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument(
        "--decoder_hidden", type=int, nargs="+", default=[512, 256, 128]
    )
    parser.add_argument("--wm_window", type=int, default=8)
    parser.add_argument("--wm_batches", type=int, default=4)
    parser.add_argument("--wm_batch", type=int, default=512)
    parser.add_argument("--pi_batch", type=int, default=512)
    parser.add_argument("--wm_only_iters", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_decay_iters", type=int, default=50_000)
    parser.add_argument("--fixed_lr", type=float, default=None)
    parser.add_argument("--retention_data", type=Path, default=None)
    parser.add_argument("--retention_weight", type=float, default=0.1)
    parser.add_argument("--retention_batch", type=int, default=1024)
    parser.add_argument("--retention_budget", type=float, default=None, help="per-group fixed-history action RMSE limit; requires a v2 bank")
    parser.add_argument("--real_critic_weight", type=float, default=0.0)
    parser.add_argument("--real_critic_batch", type=int, default=512)
    parser.add_argument("--real_critic_window", type=int, default=32)
    parser.add_argument("--checkpoint_eval_script", type=Path, default=None)
    parser.add_argument("--eval_replay_cpu", action="store_true", help="temporarily offload replay while checkpoint evaluation uses the GPU")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--action_clip", type=float, default=3.0)
    parser.add_argument("--critic_epochs", type=int, default=2)
    parser.add_argument("--critic_minibatches", type=int, default=4)
    parser.add_argument("--command_vx", type=float, nargs=2, default=[-0.5, 1.0])
    parser.add_argument("--command_vy", type=float, nargs=2, default=[-0.4, 0.4])
    parser.add_argument("--command_yaw", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--command_steps", type=int, nargs=2, default=[100, 250])
    parser.add_argument("--standing_probability", type=float, default=0.15)
    parser.add_argument("--curriculum_iters", type=int, default=10_000)
    parser.add_argument("--episode_steps", type=int, default=1000)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compile dynamics, actor rollout steps, and reward heads",
    )
    parser.add_argument("--log_dir", default="logs/go2_velocity")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None)
    initialization.add_argument(
        "--init_flat",
        default=None,
        help="initialize terrain networks from a flat Go2 checkpoint",
    )
    parser.add_argument(
        "--terrain_course", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--stairs_start", type=int, default=4000, help="start introducing 6 cm stairs"
    )
    parser.add_argument(
        "--tall_stairs_start",
        type=int,
        default=9000,
        help="start introducing 9 cm stairs",
    )
    parser.add_argument("--rough_start", type=int, default=None, help="iteration to introduce 2D rough tiles")
    parser.add_argument("--rough_ramp", type=int, default=3000, help="rough curriculum ramp length in iterations")
    parser.add_argument("--rough_seed", type=int, default=71, help="first of four training-only box geometry seeds")
    parser.add_argument(
        "--rough_route_start", type=int, default=None,
        help="start symmetric stair direction rewards; introduce 12 cm stairs after 2000 iterations",
    )
    parser.add_argument(
        "--rough_l5_focus_start", type=int, default=None,
        help="start a 4000-iteration ramp for symmetric L5 stair sampling, yaw, and speed",
    )
    parser.add_argument(
        "--rough_l5_coverage_start", type=int, default=None,
        help="start a 2000-iteration L5 yaw/spawn ramp with original terrain and speed sampling",
    )
    parser.add_argument(
        "--asset",
        default=None,
        help="optional local Go2 USD; defaults to Isaac Lab's Go2 asset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=1,
        help="CPU intra-op threads for GPU training",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="log CUDA stage timings, memory, and process CPU usage",
    )
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "supermimic-go2-v1":
            parser.error("--resume requires a Go2 checkpoint")
        # Resume inherits original settings; explicit command-line flags still take priority.
        allowed = {action.dest for action in parser._actions}
        parser.set_defaults(
            **{
                key: value
                for key, value in checkpoint["args"].items()
                if key in allowed
            }
        )
        args = parser.parse_args(argv)
    positive = (
        "num_envs",
        "iterations",
        "buffer_steps",
        "horizon",
        "hist",
        "latent",
        "wm_window",
        "wm_batches",
        "wm_batch",
        "pi_batch",
        "lr",
        "lr_decay_iters",
        "critic_epochs",
        "critic_minibatches",
        "log_every",
        "ckpt_every",
        "torch_threads",
        "rough_ramp",
        "retention_batch",
        "real_critic_batch",
        "real_critic_window",
    )
    if any(getattr(args, key) <= 0 for key in positive) or args.utd <= 0:
        parser.error(
            "Batch sizes, iteration counts, learning rate, and UTD must be positive"
        )
    if args.fixed_lr is not None and (
        not math.isfinite(args.fixed_lr) or args.fixed_lr <= 0
    ):
        parser.error("--fixed_lr must be finite and positive")
    if not math.isfinite(args.retention_weight) or args.retention_weight <= 0:
        parser.error("--retention_weight must be finite and positive")
    if args.retention_data is not None and (not args.terrain_course or not args.resume):
        parser.error("--retention_data requires a terrain checkpoint via --resume")
    if args.retention_budget is not None and (not math.isfinite(args.retention_budget) or args.retention_budget <= 0 or args.retention_data is None):
        parser.error("--retention_budget requires a finite positive budget and a reference bank")
    if not math.isfinite(args.real_critic_weight) or args.real_critic_weight < 0:
        parser.error("--real_critic_weight must be finite and nonnegative")
    if args.real_critic_weight and (args.rough_route_start is None or args.real_critic_window > args.buffer_steps or args.real_critic_batch < args.critic_minibatches):
        parser.error("Real critic targets require guided Rough terrain, a full replay window, and enough minibatch samples")
    if args.checkpoint_eval_script is not None and not args.checkpoint_eval_script.is_file():
        parser.error("--checkpoint_eval_script must point to an existing script")
    if (
        args.buffer_steps < args.wm_window
        or min(args.warmup_steps, args.wm_only_iters, args.action_noise) < 0
    ):
        parser.error(
            "Replay must fit wm_window; warmup and action noise must be nonnegative"
        )
    if not 0 < args.gamma < 1 or not 0 <= args.lam <= 1:
        parser.error("Require 0 < gamma < 1 and 0 <= lam <= 1")
    if args.init_flat and not args.terrain_course:
        parser.error("--init_flat requires --terrain_course")
    if not 0 <= args.stairs_start <= args.tall_stairs_start:
        parser.error("Require 0 <= stairs_start <= tall_stairs_start")
    if args.rough_start is not None and (
        args.rough_start < 0 or not args.terrain_course or not args.resume
    ):
        parser.error("--rough_start requires a nonnegative iteration and --resume from terrain training")
    if not 0 <= args.rough_seed < 2**32 - 3:
        parser.error("--rough_seed must allow four consecutive uint32 seeds")
    if args.rough_route_start is not None and (
        args.rough_start is None or args.rough_route_start < args.rough_start
    ):
        parser.error("--rough_route_start must be at or after --rough_start")
    if args.rough_l5_focus_start is not None and (
        args.rough_route_start is None
        or args.rough_l5_focus_start < max(
            args.rough_start + args.rough_ramp, args.rough_route_start + 5000
        )
    ):
        parser.error("--rough_l5_focus_start requires a fully ramped Rough stair curriculum")
    if args.rough_l5_coverage_start is not None and (
        args.rough_l5_focus_start is not None
        or args.rough_route_start is None
        or args.rough_l5_coverage_start < args.rough_route_start + 5000
    ):
        parser.error("--rough_l5_coverage_start requires mature Rough stairs and an independent coverage ramp")
    return args


def make_configs(args):
    network = NetworkCfg(
        hist=args.hist,
        latent=args.latent,
        wm_hidden=tuple(args.wm_hidden),
        encoder_hidden=tuple(args.encoder_hidden),
        decoder_hidden=tuple(args.decoder_hidden),
    )
    command = CommandCfg(
        vx=tuple(args.command_vx),
        vy=tuple(args.command_vy),
        yaw=tuple(args.command_yaw),
        min_steps=args.command_steps[0],
        max_steps=args.command_steps[1],
        standing_probability=args.standing_probability,
        curriculum_iters=args.curriculum_iters,
    )
    task = TaskCfg(
        action_clip=args.action_clip,
        episode_steps=args.episode_steps,
        terrain_course=args.terrain_course,
    )
    return network, command, task


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    network_cfg, command_cfg, task_cfg = make_configs(args)
    device = args.device or "cuda:0"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Go2 physics training requires a CUDA device; tensor tests can run on CPU"
        )
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    directory = Path(args.log_dir)
    if (directory / "args.json").exists() and not args.resume:
        raise FileExistsError(
            f"{directory} already contains a run; use --resume or a different --log_dir"
        )
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=True)
        if args.resume
        else None
    )
    initial = (
        torch.load(args.init_flat, map_location=device, weights_only=True)
        if args.init_flat and checkpoint is None
        else None
    )
    if checkpoint:
        for key, value in (
            ("network_cfg", asdict(network_cfg)),
            ("command_cfg", asdict(command_cfg)),
            ("task_cfg", asdict(task_cfg)),
        ):
            stored = (
                asdict(TaskCfg(**checkpoint[key]))
                if key == "task_cfg"
                else checkpoint[key]
            )
            if stored != value:
                raise ValueError(
                    f"Resume changed {key}; use the original architecture/task/command settings"
                )
        if args.iterations <= checkpoint["iteration"]:
            raise ValueError("--iterations must exceed the checkpoint iteration")
    from isaaclab.app import launch_simulation

    from .buffer import ReplayBuffer
    from .env import VelocitySim, make_sim
    from .learner import Learner

    sim_cfg = make_sim(device, terrain_course=task_cfg.terrain_course)
    route_guidance = args.rough_route_start is not None
    with launch_simulation(sim_cfg, args):
        rough_curriculum = None
        course_rates = ["course_success"]
        if args.rough_start is not None:
            from .rough_curriculum import RoughCurriculum

            rough_curriculum = RoughCurriculum(
                device, args.rough_start, args.rough_ramp,
                seed=args.rough_seed, route_start=args.rough_route_start,
                l5_focus_start=args.rough_l5_focus_start,
                l5_coverage_start=args.rough_l5_coverage_start,
            )
            course_rates += ["rough_legacy_success", *rough_curriculum.rate_keys]
            course_rates.extend(rough_curriculum.level_rate_keys.values())
        env = VelocitySim(
            sim_cfg,
            args.num_envs,
            task_cfg,
            command_cfg,
            asset=args.asset,
            stairs_start=args.stairs_start,
            tall_stairs_start=args.tall_stairs_start,
            rough_curriculum=rough_curriculum,
        )
        retention = None
        if args.retention_data is not None:
            from .retention import RetentionData

            retention = RetentionData(args.retention_data, device, checkpoint)
            if args.retention_budget is not None and not retention.guard_groups:
                raise ValueError("--retention_budget requires a v2 reference bank with disjoint guard episodes")
        learner = Learner(
            env.mdp,
            network_cfg,
            command_cfg,
            device,
            lr=args.lr,
            gamma=args.gamma,
            lam=args.lam,
            compile_model=args.compile,
            terrain=env.terrain,
            route_guidance=route_guidance,
            retention=retention,
            retention_weight=args.retention_weight,
            retention_batch=args.retention_batch,
            retention_budget=args.retention_budget,
            real_critic_weight=args.real_critic_weight,
            real_critic_batch=args.real_critic_batch,
            real_critic_window=args.real_critic_window,
        )
        buffer = ReplayBuffer(
            args.buffer_steps,
            args.num_envs,
            device,
            terrain_course=task_cfg.terrain_course,
            route_guidance=route_guidance,
            real_targets=args.real_critic_weight > 0,
        )
        history = None
        start_iteration, sim_steps, wall_offset = 0, 0, 0.0
        if checkpoint:
            if checkpoint["joint_names"] != env.joint_names:
                raise ValueError("Go2 joint ordering differs from the checkpoint")
            learner.restore(checkpoint)
            start_iteration, sim_steps, wall_offset = (
                checkpoint["iteration"],
                checkpoint["sim_steps"],
                checkpoint["wall_time"],
            )
            torch.set_rng_state(checkpoint["rng_cpu"].cpu())
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["rng_cuda"]]
            )
        elif initial is not None:
            if initial["joint_names"] != env.joint_names:
                raise ValueError(
                    "Go2 joint ordering differs from the initialization checkpoint"
                )
            learner.initialize_from_flat(initial)
            print(
                f"Initialized terrain weights from {args.init_flat} (flat iteration {initial['iteration']}); fresh optimizers.",
                flush=True,
            )
            del initial
        saved_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        (directory / "args.json").write_text(json.dumps(saved_args, indent=2) + "\n")
        (directory / "process.json").write_text(
            json.dumps({"pid": os.getpid(), "device": device}) + "\n"
        )
        (directory / "robot.json").write_text(
            json.dumps(
                {
                    "joint_names": env.joint_names,
                    "default_q": env.default_q.tolist(),
                    "lower": env.lower.tolist(),
                    "upper": env.upper.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        started = time.monotonic() - wall_offset
        stats = {}
        terrain_count_fields = []
        terrain_counts = None
        if route_guidance:
            terrain_count_fields = [
                f"{group}_{count}"
                for group in rough_curriculum.sample_groups
                for count in ("episodes", "steps")
            ]
            terrain_counts = torch.zeros(
                (len(rough_curriculum.sample_groups), 2), device=device, dtype=torch.int32
            )
            step_counts = torch.ones(env.N, device=device, dtype=torch.bool)

        def record(key, value):
            # Keep scalar statistics on the device until the logging boundary.
            stats.setdefault(key, []).append(value.detach())

        @torch.no_grad()
        def collect(steps):
            nonlocal history, sim_steps
            for _ in range(steps):
                obs = env.mdp.add_noise(env.observe())
                if history is None:
                    history = obs[:, None].repeat(1, network_cfg.hist, 1)
                else:
                    history = torch.where(
                        (env.ep_step == 0)[:, None, None],
                        obs[:, None].expand_as(history),
                        torch.cat((history[:, 1:], obs[:, None]), 1),
                    )
                action = learner.pi(history, env.command)
                action = (action + args.action_noise * torch.randn_like(action)).clamp(
                    -task_cfg.action_clip, task_cfg.action_clip
                )
                transition, metrics = env.step(action, obs)
                buffer.add(transition)
                learner.observe_transition(transition)
                if terrain_counts is not None:
                    groups = rough_curriculum.sample_group_ids[transition["terrain_id"]]
                    counts = torch.stack((transition["ep_step"] == 0, step_counts), dim=-1)
                    terrain_counts.index_add_(0, groups, counts.to(dtype=terrain_counts.dtype))
                sim_steps += args.num_envs
                record("sim_reward", transition["r"].mean())
                record("fall_fraction", transition["term"].float().mean())
                for key, value in metrics.items():
                    record(
                        key,
                        value.float()
                        if key in course_rates
                        else value.float().mean(),
                    )

        env.command_scale = (
            1.0 if task_cfg.terrain_course else command_cfg.scale(start_iteration)
        )
        if task_cfg.terrain_course and start_iteration:
            env.terrain_iteration = start_iteration
            env.reset(torch.arange(env.N, device=device))
        # Checkpoints intentionally omit physics and replay. Refill on both fresh and resumed launches.
        collect(max(args.warmup_steps, args.wm_window))
        if retention is not None and args.real_critic_weight:
            from .real_value import validate_real_values
            path = directory / f"real_value_validation_{start_iteration:06d}.json"
            if not path.exists():
                path.write_text(json.dumps(validate_real_values(learner, retention), indent=2, allow_nan=False) + "\n")
        fields = [
            "iteration",
            "wall_h",
            "sim_steps",
            "wm_loss",
            "wm_last_error",
            "pi_loss",
            "v_loss",
            "pi_gradnorm",
            "imag_reward",
            "imag_survival",
            "sim_reward",
            "linear_error",
            "yaw_error",
            "fall_fraction",
            "episode_return",
            "episode_length",
            "command_scale",
            "it_per_s",
            "lr",
        ]
        if task_cfg.terrain_course:
            fields += [
                "contact_fraction",
                "wm_contact_loss",
                "wm_contact_accuracy",
                "stairs_fraction",
                "course_success",
            ]
        if rough_curriculum is not None:
            fields += ["rough_fraction", *course_rates[1:]]
        if retention is not None:
            fields += ["retention_loss", "retention_rmse"]
        if args.retention_budget is not None:
            fields += ["retention_weight", "retention_worst_rmse", "retention_guard_rmse", "retention_grad_ratio", "pi_step_scale", "pi_step_rejected"]
        if args.real_critic_weight:
            fields += ["real_v_loss", "real_v_bias", "real_v_terminal_fraction"]
        fields += terrain_count_fields
        if args.profile:
            fields += [
                "collect_ms",
                "world_model_ms",
                "policy_critic_ms",
                "cpu_percent",
                "cuda_allocated_gib",
                "cuda_reserved_gib",
                "cuda_peak_allocated_gib",
                "imagined_steps_per_s",
                "sim_transitions_per_s",
            ]
        with (directory / "train_log.csv").open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if stream.tell() == 0:
                writer.writeheader()
            else:
                with (directory / "train_log.csv").open() as existing:
                    if next(csv.reader(existing)) != fields:
                        raise ValueError(
                            "Log columns changed; resume into a new --log_dir when toggling --profile"
                        )
            step_accumulator, last_log, logged_at = (
                0.0,
                time.monotonic(),
                start_iteration,
            )
            phase_events = []
            usage = resource.getrusage(resource.RUSAGE_SELF)
            last_cpu = usage.ru_utime + usage.ru_stime
            measured_sim_steps = sim_steps
            for iteration in range(start_iteration, args.iterations):
                lr = args.lr * (
                    0.03
                    + 0.97
                    * 0.5
                    * (1 + math.cos(math.pi * min(iteration / args.lr_decay_iters, 1)))
                )
                if args.fixed_lr is not None:
                    lr = args.fixed_lr
                for optimizer in (learner.opt_wm, learner.opt_pi, learner.opt_v):
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                env.command_scale = (
                    1.0 if task_cfg.terrain_course else command_cfg.scale(iteration)
                )
                env.terrain_iteration = iteration
                step_accumulator += 1 / args.utd
                steps = int(step_accumulator)
                step_accumulator -= steps
                if args.profile:
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    events[0].record()
                collect(steps)
                if args.profile:
                    events[1].record()
                for _ in range(args.wm_batches):
                    for key, value in learner.wm_update(
                        buffer, args.wm_batch, args.wm_window
                    ).items():
                        record(key, value)
                if args.profile:
                    events[2].record()
                if iteration >= args.wm_only_iters:
                    update = learner.policy_update(
                        buffer,
                        args.pi_batch,
                        args.horizon,
                        command_scale=env.command_scale,
                        critic_epochs=args.critic_epochs,
                        minibatches=args.critic_minibatches,
                        diagnostics=args.retention_budget is not None and iteration % args.log_every == 0,
                    )
                    for key, value in update.items():
                        record(key, value)
                if args.profile:
                    events[3].record()
                    phase_events.append(events)
                number = iteration + 1
                if number % args.log_every == 0 or number == args.iterations:
                    keys = list(stats)
                    means = (
                        torch.stack(
                            [
                                (
                                    torch.cat(stats[key])
                                    if key in course_rates
                                    else torch.stack(stats[key])
                                ).mean()
                                for key in keys
                            ]
                        )
                        .cpu()
                        .tolist()
                    )
                    if args.profile:
                        torch.cuda.synchronize()
                    now = time.monotonic()
                    row = dict(zip(keys, means, strict=True))
                    if terrain_counts is not None:
                        row.update(zip(terrain_count_fields, terrain_counts.flatten().tolist(), strict=True))
                        terrain_counts.zero_()
                    row.update(
                        iteration=number,
                        wall_h=(now - started) / 3600,
                        sim_steps=sim_steps,
                        command_scale=env.command_scale,
                        it_per_s=(number - logged_at) / (now - last_log),
                        lr=lr,
                    )
                    if args.profile:
                        usage = resource.getrusage(resource.RUSAGE_SELF)
                        cpu = usage.ru_utime + usage.ru_stime
                        elapsed = now - last_log
                        count = len(phase_events)
                        phase_names = (
                            "collect_ms",
                            "world_model_ms",
                            "policy_critic_ms",
                        )
                        for index, name in enumerate(phase_names):
                            row[name] = (
                                sum(
                                    e[index].elapsed_time(e[index + 1])
                                    for e in phase_events
                                )
                                / count
                            )
                        row.update(
                            cpu_percent=100 * (cpu - last_cpu) / elapsed,
                            cuda_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                            cuda_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()
                            / 2**30,
                            imagined_steps_per_s=max(
                                0, number - max(logged_at, args.wm_only_iters)
                            )
                            * args.pi_batch
                            * args.horizon
                            / elapsed,
                            sim_transitions_per_s=(sim_steps - measured_sim_steps)
                            / elapsed,
                        )
                        last_cpu, measured_sim_steps = cpu, sim_steps
                        phase_events.clear()
                    writer.writerow(row)
                    stream.flush()
                    print(
                        f"it {number} | wm {row.get('wm_loss', 0):.4f} | sim_r {row.get('sim_reward', 0):.4f} "
                        f"| v_err {row.get('linear_error', 0):.3f} m/s | yaw_err {row.get('yaw_error', 0):.3f} rad/s "
                        f"| imag_surv {row.get('imag_survival', 0):.3f} | {row['it_per_s']:.2f} it/s",
                        flush=True,
                    )
                    if args.profile:
                        print(
                            f"  collect {row['collect_ms']:.0f} ms | WM {row['world_model_ms']:.0f} ms "
                            f"| policy+critic {row['policy_critic_ms']:.0f} ms | CPU {row['cpu_percent']:.0f}% "
                            f"| torch peak {row['cuda_peak_allocated_gib']:.2f} GiB",
                            flush=True,
                        )
                    stats.clear()
                    last_log, logged_at = now, number
                if number % args.ckpt_every == 0 or number == args.iterations:
                    saved = learner.checkpoint()
                    saved.update(
                        args=saved_args,
                        iteration=number,
                        sim_steps=sim_steps,
                        wall_time=time.monotonic() - started,
                        joint_names=env.joint_names,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all(),
                    )
                    atomic_save(saved, directory / "last.pt")
                    atomic_save(saved, directory / f"ckpt_{number:06d}.pt")
                    if retention is not None:
                        validation = retention.validate(learner.wm, env.terrain)
                        (directory / f"wm_validation_{number:06d}.json").write_text(
                            json.dumps(validation, indent=2, allow_nan=False) + "\n"
                        )
                        if args.retention_budget is not None:
                            (directory / f"policy_retention_validation_{number:06d}.json").write_text(json.dumps(retention.measure_policy(learner.pi), indent=2, allow_nan=False) + "\n")
                        if args.real_critic_weight:
                            (directory / f"real_value_validation_{number:06d}.json").write_text(json.dumps(validate_real_values(learner, retention), indent=2, allow_nan=False) + "\n")
                    if args.checkpoint_eval_script is not None:
                        torch.cuda.synchronize()
                        if args.eval_replay_cpu:
                            replay_bytes = sum(v.numel() * v.element_size() for v in buffer.data.values())
                            print(f"Offloading {replay_bytes / 2**30:.2f} GiB of replay for evaluation", flush=True)
                            buffer.move_("cpu")
                        torch.cuda.empty_cache()
                        try:
                            evaluation = subprocess.run(
                                [
                                    sys.executable, str(args.checkpoint_eval_script.resolve()),
                                    "--ckpt", str((directory / f"ckpt_{number:06d}.pt").resolve()),
                                    "--run_dir", str(directory.resolve()),
                                ],
                                check=False,
                            )
                        finally:
                            if args.eval_replay_cpu:
                                buffer.move_(device)
                        if evaluation.returncode == 23:
                            print("Evaluation regression gate stopped continuation; saved checkpoint retained.", flush=True)
                            break
                        evaluation.check_returncode()


if __name__ == "__main__":
    main()
