"""Serializable settings; this module does not import the simulator."""

from dataclasses import dataclass

NUM_J = 12
NUM_FEET = 4
STATE_DIM = 13 + 2 * NUM_J  # p, quaternion wxyz, world v/w, q, qd
OBS_DIM = 3 * NUM_J + 6
INV_DIM = 13 + 2 * NUM_J
XI_DIM = 1 + NUM_J  # foot friction and joint calibration offsets
WM_OUT = 12 + 2 * NUM_J
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4


@dataclass
class CommandCfg:
    vx: tuple[float, float] = (-0.5, 1.0)
    vy: tuple[float, float] = (-0.4, 0.4)
    yaw: tuple[float, float] = (-1.0, 1.0)
    min_steps: int = 100
    max_steps: int = 250
    standing_probability: float = 0.15
    curriculum_iters: int = 10_000
    initial_scale: float = 0.2

    def __post_init__(self):
        if self.min_steps < 1 or self.max_steps < self.min_steps:
            raise ValueError(
                "Command durations must satisfy 1 <= min_steps <= max_steps"
            )
        if not 0 <= self.standing_probability <= 1 or not 0 < self.initial_scale <= 1:
            raise ValueError("Invalid standing probability or initial command scale")
        if self.curriculum_iters < 0:
            raise ValueError("curriculum_iters must be nonnegative")
        if any(len(r) != 2 or r[0] > r[1] for r in (self.vx, self.vy, self.yaw)):
            raise ValueError("Command ranges must contain ordered lower/upper bounds")

    def scale(self, iteration: int) -> float:
        fraction = (
            1.0
            if self.curriculum_iters == 0
            else min(max(iteration, 0) / self.curriculum_iters, 1.0)
        )
        return self.initial_scale + (1 - self.initial_scale) * fraction


@dataclass
class TaskCfg:
    action_scale: float = 0.25
    action_clip: float = 3.0
    episode_steps: int = 1000
    target_height: float = 0.34
    min_height: float = 0.16
    min_upright: float = 0.5
    linear_std: float = 0.5
    angular_std: float = 0.5
    w_linear: float = 1.5
    w_angular: float = 0.75
    w_vertical: float = 2.0
    w_roll_pitch_rate: float = 0.05
    w_orientation: float = 2.5
    w_height: float = 10.0
    w_action_rate: float = 0.005
    w_joint_limit: float = 2.0
    w_posture: float = 0.02
    w_standing: float = 0.5
    terrain_course: bool = False

    def __post_init__(self):
        if (
            min(self.action_scale, self.action_clip, self.linear_std, self.angular_std)
            <= 0
        ):
            raise ValueError(
                "Action scales and tracking standard deviations must be positive"
            )
        if self.episode_steps < 1 or not 0 < self.min_height < self.target_height:
            raise ValueError("Invalid episode length or standing/fall heights")
        if not -1 < self.min_upright < 1:
            raise ValueError("min_upright must be a cosine threshold in (-1, 1)")


@dataclass
class NetworkCfg:
    hist: int = 10
    latent: int = 32
    wm_hidden: tuple[int, ...] = (512, 512, 512)
    encoder_hidden: tuple[int, ...] = (256, 256)
    decoder_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (256, 128)

    def __post_init__(self):
        if min(self.hist, self.latent) < 1:
            raise ValueError("History and latent dimensions must be positive")
        for widths in (
            self.wm_hidden,
            self.encoder_hidden,
            self.decoder_hidden,
            self.critic_hidden,
        ):
            if not widths or min(widths) < 1:
                raise ValueError("Network hidden widths must be positive")
