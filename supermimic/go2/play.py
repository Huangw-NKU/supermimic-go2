"""Run a trained Go2 with keyboard velocity commands in a Newton GL window."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from .config import CONTROL_DT, CommandCfg, NetworkCfg, TaskCfg
from .networks import VelocityPolicy
from .teleop import KeyboardCommands
from .terrain import FOOT_NAMES, MIXED_TILE, add_terrain_args, resolve_terrain_tile

REPO = Path(__file__).resolve().parents[2]
MIXED_START_VX = 0.5


def default_checkpoint(root=REPO):
    """Prefer a configured checkpoint, then a local active run or newest run."""
    configured = os.environ.get("SUPERMIMIC_GO2_CKPT")
    if configured:
        # Return an invalid explicit selection too, so the parser reports it
        # rather than silently loading a different model.
        return Path(configured).expanduser()
    active = root / ".cache/active_go2_run.json"
    if active.is_file():
        try:
            candidate = Path(json.loads(active.read_text())["run_dir"]) / "last.pt"
            if candidate.is_file():
                return candidate
        except (KeyError, ValueError, TypeError):
            pass
    candidates = list((root / "logs").glob("*/last.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


class NewtonKeyboard:
    """Bind Pyglet events ahead of Newton's camera/pause keyboard handlers."""

    def __init__(self, visualizer, commands, iteration):
        # The native window is the only private Isaac Lab bridge used here.
        # It has no public key-event registration API; keep access in one adapter.
        from pyglet.event import EVENT_HANDLED
        from pyglet.window import key

        self.viewer = getattr(visualizer, "_viewer", None)
        if self.viewer is None:
            raise RuntimeError("Keyboard playback requires the Newton GL visualizer")
        self.window = self.viewer.renderer.window
        self.commands, self.iteration = commands, iteration
        self.actual_vx = self.actual_yaw = 0.0
        self.course_progress = None
        self.foot_contacts = None
        self.ghost_steps = 0
        self.resets = 0
        self.pressed = set()
        self._handled = EVENT_HANDLED
        self.bindings = {
            key.W: "forward",
            key.UP: "forward",
            key.S: "backward",
            key.DOWN: "backward",
            key.A: "left",
            key.LEFT: "left",
            key.D: "right",
            key.RIGHT: "right",
            key.SPACE: "stop",
            key.R: "reset",
            key.F: "follow",
            key.Q: "quit",
            key.ESCAPE: "quit",
        }
        self.window.push_handlers(self)
        self.window.set_caption(f"Go2 keyboard control | checkpoint {iteration}")
        self.viewer.register_ui_callback(self.draw_hud, position="side")

    def on_key_press(self, symbol, modifiers):
        action = self.bindings.get(symbol)
        if action is None:
            return None
        if symbol in self.pressed:
            return self._handled
        self.pressed.add(symbol)
        self.commands.press(action)
        if action == "quit":
            # Also closes a viewer paused by the native Simulation controls.
            self.viewer.request_close()
        return self._handled

    def on_key_release(self, symbol, modifiers):
        action = self.bindings.get(symbol)
        if action is None:
            return None
        self.pressed.discard(symbol)
        if not any(self.bindings.get(held) == action for held in self.pressed):
            self.commands.release(action)
        return self._handled

    def on_deactivate(self):
        self.pressed.clear()
        self.commands.stop()

    def draw_hud(self, imgui):
        imgui.separator()
        imgui.text("Go2 keyboard velocity control")
        imgui.text(f"Checkpoint: {self.iteration} | resets: {self.resets}")
        if self.course_progress is not None:
            imgui.text(self.course_progress)
        if self.ghost_steps:
            imgui.text_colored(
                (0.30, 0.85, 1.0, 1.0),
                f"Cyan: world model +{self.ghost_steps * CONTROL_DT:.2f} s",
            )
        imgui.text(
            f"Target  vx: {self.commands.vx:+.2f} m/s   yaw: {self.commands.yaw:+.2f} rad/s"
        )
        imgui.text(
            f"Actual  vx: {self.actual_vx:+.2f} m/s   yaw: {self.actual_yaw:+.2f} rad/s"
        )
        if self.foot_contacts is not None:
            imgui.text("Foot contact (1 = contact)")
            flags = imgui.ColorEditFlags_.no_tooltip | imgui.ColorEditFlags_.no_drag_drop
            for index, (foot, contact) in enumerate(zip(FOOT_NAMES, self.foot_contacts)):
                leg = foot.removesuffix("_foot")
                color = (0.18, 0.84, 0.44, 1.0) if contact else (0.43, 0.49, 0.58, 1.0)
                if index % 2:
                    imgui.same_line(94)
                imgui.color_button(f"##{leg}_contact", color, flags, (26, 17))
                imgui.same_line()
                imgui.text(f"{leg}  {contact}")
            imgui.separator()
        imgui.text("W/S or Up/Down: change forward velocity")
        imgui.text("A/D or Left/Right: change yaw rate")
        imgui.text("Release: hold target | Space: zero target")
        imgui.text("R: reset | F: follow camera | Q/Esc: quit")
        imgui.text("Focus loss zeros the velocity target")
        if imgui.button("Zero velocity"):
            self.commands.stop()
        imgui.same_line()
        if imgui.button("Reset Go2"):
            self.commands.request_reset()

    def close(self):
        if self.window.context is not None:
            self.window.remove_handlers(self)


@torch.no_grad()
def run_session(
    env, policy, hist, commands, *, keyboard=None, ghost=None, max_steps=None, real_time=True
):
    """Single-robot policy loop; supports bounded headless integration checks."""
    history, steps, resets = None, 0, 0
    sections = env.terrain.sections if env.terrain is not None else ()
    initial_vx, finished = commands.vx, False
    ids = torch.arange(1, device=env.device)
    while (
        env.sim.is_running()
        and not commands.quit_requested
        and (max_steps is None or steps < max_steps)
    ):
        started = time.perf_counter()
        native_reset = any(v.consume_reset_request() for v in env.sim.visualizers)
        keyboard_reset = commands.consume_reset()
        if native_reset or keyboard_reset:
            commands.stop()
            if sections:
                commands.vx = initial_vx
                finished = False
            env.reset(ids)
            history = None
            if ghost is not None:
                ghost.ready = False
            resets += 1
        command = commands.update(CONTROL_DT)
        env.command[:] = env.command.new_tensor(command)
        env.remaining.fill_(env.command_cfg.max_steps)
        state = env.state()
        obs = env.observe(state)
        history = (
            obs[:, None].repeat(1, hist, 1)
            if history is None
            else torch.cat((history[:, 1:], obs[:, None]), 1)
        )
        transition, _ = env.step(policy(history, env.command), obs)
        if not torch.isfinite(transition["s1"]).all():
            raise FloatingPointError("Go2 playback produced a nonfinite physical state")
        fell = transition["term"].item()
        following = env.state()
        if sections and not finished and transition["course_reached"].item():
            finished = True
            commands.stop()
            print(f"Mixed course complete at x={env.terrain.finish_x:.2f} m.", flush=True)
        if keyboard is not None:
            measured = env.mdp.velocity(following)[0].cpu().tolist()
            keyboard.actual_vx, keyboard.actual_yaw = measured[0], measured[2]
            keyboard.resets = resets
            if env.contacts is not None:
                keyboard.foot_contacts = obs[0, -len(FOOT_NAMES) :].int().tolist()
            if sections:
                x = following[0][0, 0].item()
                section = next(
                    (
                        f"{i}/{len(sections)} {name}"
                        for i, (name, end) in enumerate(sections, 1)
                        if x < end
                    ),
                    "Finish",
                )
                if finished:
                    section = "Complete"
                keyboard.course_progress = (
                    f"{section} | {x:.1f} / {env.terrain.finish_x:.1f} m"
                )
        if env.sim.is_rendering:
            if ghost is not None and ghost.is_visible():
                if fell:
                    ghost.ready = False
                elif steps % 2 == 0 or not ghost.ready:
                    next_history = torch.cat(
                        (history[:, 1:], env.observe(following)[:, None]), dim=1
                    )
                    ghost.forecast(policy, following, next_history)
            if commands.follow_camera:
                target = (following[0][0] + env.origins[0]).cpu().tolist()
                eye = [target[0] + 2.0, target[1] - 2.0, target[2] + 1.2]
                if env.terrain is not None:
                    eye = [target[0] - 1.9, target[1] - 2.6, target[2] + 1.45]
                    target[0] += 0.55
                    target[2] -= 0.12
                env.sim.set_camera_view(
                    eye=eye,
                    target=target,
                )
            env.sim.render()
        if fell:
            print("Go2 fell; resetting to standing and zeroing the target.", flush=True)
            commands.stop()
            env.reset(ids)
            history = None
            finished = False
            resets += 1
        # Evaluation timeouts are ignored so keyboard playback can continue beyond 20 s.
        steps += 1
        if real_time:
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - started)))
    return {"steps": steps, "resets": resets}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="defaults to the current Go2 run's last.pt",
    )
    parser.add_argument("--asset", default=None, help="optional local Go2 USD")
    add_terrain_args(parser)
    parser.add_argument(
        "--start_vx",
        type=float,
        default=None,
        help=f"initial forward target in m/s (mixed: {MIXED_START_VX}, other lanes: zero)",
    )
    parser.add_argument(
        "--ghost_steps",
        type=int,
        default=0,
        help="world-model forecast horizon in 20 ms steps; zero hides the ghost",
    )
    parser.add_argument(
        "--linear_accel",
        type=float,
        default=0.5,
        help="W/S target change rate in m/s^2",
    )
    parser.add_argument(
        "--angular_accel",
        type=float,
        default=1.0,
        help="A/D target change rate in rad/s^2",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="optional time limit; zero runs until the window closes",
    )
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton_gl")
    args = parser.parse_args(argv)
    args.terrain_tile = resolve_terrain_tile(parser, args)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if args.start_vx is not None and not math.isfinite(args.start_vx):
        parser.error("--start_vx must be finite")
    if args.ghost_steps < 0:
        parser.error("--ghost_steps must be nonnegative")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.linear_accel, args.angular_accel)
    ):
        parser.error("Keyboard accelerations must be finite and positive")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error(
            "Keyboard playback needs a local desktop display; run this command in your desktop terminal"
        )
    # Isaac Lab normalizes both defaults and explicit flags into a list.
    # Accept the older string form too, then pass the launcher's native form on.
    selected = (
        [args.visualizer] if isinstance(args.visualizer, str) else args.visualizer
    )
    if selected not in (["newton_gl"], ["newton"]):
        parser.error("Keyboard playback requires --visualizer newton_gl")
    args.visualizer = ["newton_gl"]
    path = args.ckpt or default_checkpoint()
    if path is None or not path.is_file():
        parser.error(
            "No trained Go2 checkpoint found; specify --ckpt /absolute/path/to/checkpoint.pt"
        )
    args.ckpt = path
    return args


def main(argv=None):
    args = parse_args(argv)
    path = args.ckpt
    torch.set_num_threads(1)
    device = args.device or "cuda:0"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Go2 playback requires a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "supermimic-go2-v1":
        raise ValueError("Expected a Go2 checkpoint")
    net_cfg, task_cfg = (
        NetworkCfg(**checkpoint["network_cfg"]),
        TaskCfg(**checkpoint["task_cfg"]),
    )
    command_cfg = CommandCfg(**checkpoint["command_cfg"])
    start_vx = args.start_vx
    if start_vx is None:
        start_vx = MIXED_START_VX if args.terrain_tile == MIXED_TILE else 0.0
    controls = KeyboardCommands(
        tuple(command_cfg.vx),
        tuple(command_cfg.yaw),
        args.linear_accel,
        args.angular_accel,
        vx=start_vx,
    )
    if args.terrain_tile is not None and not task_cfg.terrain_course:
        raise ValueError("--terrain_tile requires a Go2 terrain checkpoint")
    policy = VelocityPolicy(
        net_cfg, task_cfg.action_clip, foot_contacts=task_cfg.terrain_course
    ).to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    from isaaclab.app import launch_simulation

    from .env import VelocitySim, make_sim

    cfg = make_sim(device, terrain_course=task_cfg.terrain_course)
    cfg.render_interval = round(CONTROL_DT / cfg.dt)
    print(
        f"Loading Go2 checkpoint {path.resolve()} (iteration {checkpoint['iteration']}).",
        flush=True,
    )
    print(
        f"vx limits: {controls.vx_bounds} m/s | yaw limits: {controls.yaw_bounds} rad/s",
        flush=True,
    )
    print(
        "Click the Newton window. Hold W/S: forward/back; A/D: turn. Release: hold target.",
        flush=True,
    )
    print(
        "Space: zero | R: reset | F: follow camera | Q/Esc: quit. Focus loss zeros the target.",
        flush=True,
    )
    with launch_simulation(cfg, args):
        env = VelocitySim(
            cfg,
            1,
            task_cfg,
            command_cfg,
            eval_mode=True,
            asset=args.asset or checkpoint["args"].get("asset"),
            terrain_tile=args.terrain_tile or "step_up",
            terrain_sequence=args.terrain_sequence,
        )
        if env.joint_names != checkpoint["joint_names"]:
            raise ValueError("Go2 joint ordering differs from the checkpoint")
        if args.terrain_tile == MIXED_TILE:
            sequence = " -> ".join(name for name, _ in env.terrain.sections)
            print(
                f"Mixed course: {env.terrain.finish_x:.2f} m. Start at {start_vx:.2f} m/s. "
                f"R restarts the course.\n{sequence}",
                flush=True,
            )
        visualizer = next(
            (
                v
                for v in env.sim.visualizers
                if v.cfg.backend.visualizer_type == "newton_gl"
            ),
            None,
        )
        if visualizer is None:
            raise RuntimeError("Newton GL window was not created")
        keyboard = NewtonKeyboard(visualizer, controls, checkpoint["iteration"])
        ghost = None
        try:
            if env.terrain is not None:
                from .terrain_view import paint_lane, show_step_dimensions

                paint_lane(visualizer, env.terrain, env.eval_tile, env.device)
                show_step_dimensions(keyboard.viewer, env.terrain, env.eval_tile)
            if args.ghost_steps:
                from .prediction_view import PredictionGhost

                ghost = PredictionGhost(env, net_cfg, checkpoint["wm"], args.ghost_steps)
                keyboard.ghost_steps = args.ghost_steps
            result = run_session(
                env,
                policy,
                net_cfg.hist,
                controls,
                keyboard=keyboard,
                ghost=ghost,
                max_steps=math.ceil(args.seconds / CONTROL_DT)
                if args.seconds
                else None,
            )
            print(
                f"Keyboard playback finished: {result['steps']} steps, {result['resets']} resets.",
                flush=True,
            )
        except KeyboardInterrupt:
            print("Keyboard playback stopped.", flush=True)
        finally:
            if ghost is not None:
                ghost.close()
            keyboard.close()


if __name__ == "__main__":
    main()
