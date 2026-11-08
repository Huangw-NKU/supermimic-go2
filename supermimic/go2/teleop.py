"""Simulator-independent keyboard command state for Go2 playback."""

import math
from dataclasses import dataclass, field


@dataclass
class KeyboardCommands:
    """Held keys change a velocity setpoint; releasing them holds that setpoint."""

    vx_bounds: tuple[float, float] = (-0.5, 1.0)
    yaw_bounds: tuple[float, float] = (-1.0, 1.0)
    linear_accel: float = 0.5
    angular_accel: float = 1.0
    vx: float = 0.0
    yaw: float = 0.0
    follow_camera: bool = True
    quit_requested: bool = False
    reset_requested: bool = False
    held: set[str] = field(default_factory=set)

    def __post_init__(self):
        for bounds in (self.vx_bounds, self.yaw_bounds):
            if (
                len(bounds) != 2
                or not all(math.isfinite(x) for x in bounds)
                or not bounds[0] <= 0 <= bounds[1]
            ):
                raise ValueError(
                    "Keyboard velocity bounds must be finite and contain zero"
                )
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.linear_accel, self.angular_accel)
        ):
            raise ValueError("Keyboard accelerations must be finite and positive")
        if (
            not self.vx_bounds[0] <= self.vx <= self.vx_bounds[1]
            or not self.yaw_bounds[0] <= self.yaw <= self.yaw_bounds[1]
        ):
            raise ValueError(
                "Initial command must be inside the checkpoint's velocity limits"
            )

    def stop(self):
        self.vx = self.yaw = 0.0
        self.held.clear()

    def request_reset(self):
        self.stop()
        self.reset_requested = True

    def press(self, key):
        if key in self.held:
            return
        if key == "stop":
            self.stop()
        elif key == "reset":
            self.request_reset()
        elif key == "quit":
            self.stop()
            self.quit_requested = True
        elif key == "follow":
            self.follow_camera = not self.follow_camera
        self.held.add(key)

    def release(self, key):
        self.held.discard(key)

    def update(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Command timestep must be finite and nonnegative")
        if "stop" in self.held or self.quit_requested:
            self.vx = self.yaw = 0.0
        else:
            forward = int("forward" in self.held) - int("backward" in self.held)
            turn = int("left" in self.held) - int("right" in self.held)
            self.vx = min(
                max(self.vx + forward * self.linear_accel * dt, self.vx_bounds[0]),
                self.vx_bounds[1],
            )
            self.yaw = min(
                max(self.yaw + turn * self.angular_accel * dt, self.yaw_bounds[0]),
                self.yaw_bounds[1],
            )
        return self.vx, 0.0, self.yaw

    def consume_reset(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested
