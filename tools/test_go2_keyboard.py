"""Keyboard command regressions; no GPU, simulator, or window required."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supermimic.go2.play import parse_args
from supermimic.go2.teleop import KeyboardCommands


class KeyboardLauncherTest(unittest.TestCase):
    def test_terrain_sequence_selects_mixed_and_keeps_repeated_sections(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"DISPLAY": ":1"}),
        ):
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            sequence = ["stairs_down", "flat", "stairs_down"]
            args = parse_args(["--ckpt", str(checkpoint), "--terrain_sequence", *sequence])
            self.assertEqual(args.terrain_tile, "mixed")
            self.assertEqual(args.terrain_sequence, sequence)

    def test_launcher_accepts_default_and_explicit_newton_visualizer(self):
        # Use the actual Isaac Lab parser: it converts visualizer strings to lists.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"DISPLAY": ":1"}),
        ):
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            for flags in (
                [],
                ["--visualizer", "newton_gl"],
                ["--visualizer", "newton"],
            ):
                with self.subTest(flags=flags):
                    args = parse_args(["--ckpt", str(checkpoint), *flags])
                    self.assertEqual(args.visualizer, ["newton_gl"])
                    self.assertEqual(args.ckpt, checkpoint)

    def test_launcher_rejects_disabled_or_multiple_visualizers(self):
        with patch.dict("os.environ", {"DISPLAY": ":1"}):
            for value in ("none", "viser", "newton_gl,viser"):
                with (
                    self.subTest(value=value),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    parse_args(["--visualizer", value])
                self.assertEqual(error.exception.code, 2)


class KeyboardCommandsTest(unittest.TestCase):
    def test_hold_changes_setpoint_and_release_keeps_it(self):
        controls = KeyboardCommands()
        controls.press("forward")
        controls.press("left")
        for _ in range(25):
            # Repeated OS key events must not alter the change rate.
            controls.press("forward")
            controls.update(0.02)
        self.assertAlmostEqual(controls.vx, 0.25)
        self.assertAlmostEqual(controls.yaw, 0.5)
        controls.release("forward")
        controls.release("left")
        self.assertEqual(controls.update(1), (controls.vx, 0.0, controls.yaw))
        self.assertAlmostEqual(controls.vx, 0.25)
        self.assertAlmostEqual(controls.yaw, 0.5)

    def test_forward_reverse_and_turn_limits_follow_checkpoint(self):
        controls = KeyboardCommands(vx_bounds=(-0.3, 0.8), yaw_bounds=(-0.7, 0.6))
        controls.press("forward")
        controls.press("left")
        self.assertEqual(controls.update(20), (0.8, 0.0, 0.6))
        controls.release("forward")
        controls.release("left")
        controls.press("backward")
        controls.press("right")
        self.assertEqual(controls.update(20), (-0.3, 0.0, -0.7))

    def test_opposing_keys_cancel_without_jumping_the_target(self):
        controls = KeyboardCommands(vx=0.2, yaw=-0.1)
        for key in ("forward", "backward", "left", "right"):
            controls.press(key)
        self.assertEqual(controls.update(1), (0.2, 0.0, -0.1))

    def test_space_and_focus_loss_clear_motion(self):
        controls = KeyboardCommands(vx=0.6, yaw=0.4)
        controls.press("forward")
        controls.press("stop")
        controls.press("forward")
        self.assertEqual(controls.update(1), (0.0, 0.0, 0.0))
        controls.release("stop")
        controls.stop()  # Window focus-loss handler uses this same operation.
        self.assertEqual(controls.update(1), (0.0, 0.0, 0.0))
        self.assertEqual(controls.held, set())

    def test_reset_is_consumed_once_and_can_be_requested_again(self):
        controls = KeyboardCommands(vx=0.5, yaw=0.2)
        controls.press("reset")
        self.assertTrue(controls.consume_reset())
        self.assertFalse(controls.consume_reset())
        self.assertEqual(controls.update(1), (0.0, 0.0, 0.0))
        controls.request_reset()  # Also exercises repeated HUD button resets.
        self.assertTrue(controls.consume_reset())

    def test_follow_toggle_ignores_repeats_and_quit_stops_motion(self):
        controls = KeyboardCommands(vx=0.5)
        controls.press("follow")
        controls.press("follow")
        self.assertFalse(controls.follow_camera)
        controls.release("follow")
        controls.press("follow")
        self.assertTrue(controls.follow_camera)
        controls.press("quit")
        self.assertTrue(controls.quit_requested)
        self.assertEqual(controls.update(1), (0.0, 0.0, 0.0))

    def test_invalid_acceleration_limits_and_timestep_are_rejected(self):
        for options in (
            {"linear_accel": 0},
            {"angular_accel": float("nan")},
            {"vx_bounds": (0.1, 1.0)},
            {"yaw_bounds": (-1.0, float("inf"))},
            {"vx": 2.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                KeyboardCommands(**options)
        for dt in (-1, float("inf"), float("nan")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                KeyboardCommands().update(dt)


if __name__ == "__main__":
    unittest.main()
