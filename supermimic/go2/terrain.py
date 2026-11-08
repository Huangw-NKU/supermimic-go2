"""Static terrain lanes and batched ground queries for Go2 imagination."""

import math

import numpy as np
import torch

TILE_NAMES = (
    "flat",
    "ramp_up",
    "ramp_down",
    "bumps",
    "step_up",
    "step_down",
    "stairs_up_low",
    "stairs_down_low",
    "stairs_up",
    "stairs_down",
)
MIXED_TILE = "mixed"
EVAL_TILE_NAMES = (*TILE_NAMES, MIXED_TILE)
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
X_MIN, X_MAX, HALF_WIDTH, LANE_SPACING = -3.0, 16.0, 2.0, 5.0
FINISH_X = 3.15
GRID_SPACING = 0.02
OFFSETS = (
    (0.22, 0.16),
    (0.22, -0.16),
    (-0.22, 0.16),
    (-0.22, -0.16),
    (0.0, 0.0),
    (0.36, 0.16),
    (0.36, -0.16),
    (0.52, 0.16),
    (0.52, -0.16),
)
TERRAIN_DIM = len(OFFSETS)


def profiles():
    def ramp(slope):
        return [(X_MIN, 0.0), (0.35, 0.0), (3.4, 3.05 * slope), (X_MAX, 3.05 * slope)]

    def stairs(height, steps):
        points = [(X_MIN, 0.0)]
        for index in range(steps):
            x = 0.65 + 0.30 * index
            points.extend(((x, index * height), (x, (index + 1) * height)))
        return points + [(X_MAX, steps * height)]

    bumps = [(X_MIN, 0.0), (0.30, 0.0)]
    for x in np.linspace(0.38, 3.75, 43):
        phase = (x - 0.30) / 3.45
        z = (
            0.025
            * math.sin(math.pi * phase)
            * math.sin(2 * math.pi * (x - 0.30) / 0.58)
        )
        bumps.append((float(x), z))
    bumps.extend(((3.76, 0.0), (X_MAX, 0.0)))
    return (
        [(X_MIN, 0.0), (X_MAX, 0.0)],
        ramp(0.10),
        ramp(-0.10),
        bumps,
        stairs(0.045, 1),
        stairs(-0.045, 1),
        stairs(0.060, 6),
        stairs(-0.060, 6),
        stairs(0.090, 6),
        stairs(-0.090, 6),
    )


def add_terrain_args(parser):
    parser.add_argument(
        "--terrain_tile",
        choices=EVAL_TILE_NAMES,
        help="course lane for a terrain checkpoint (default: step_up)",
    )
    parser.add_argument(
        "--terrain_sequence",
        nargs="+",
        choices=TILE_NAMES,
        help="ordered sections of a mixed course; repeats are allowed",
    )


def resolve_terrain_tile(parser, args):
    if args.terrain_sequence is None:
        return args.terrain_tile
    if args.terrain_tile not in (None, MIXED_TILE):
        parser.error("--terrain_sequence requires --terrain_tile mixed")
    return MIXED_TILE


def mixed_profile(lanes, sequence):
    """Join the existing obstacles with level landings at their exit heights."""
    if not sequence:
        raise ValueError("The terrain sequence needs at least one section")
    points = [(X_MIN, 0.0)]
    sections = []
    if sequence[0] != "flat":
        points.append((0.65, 0.0))
    for index, tile in enumerate(sequence):
        start_x, start_z = points[-1]
        if tile == "flat":
            points.append((round(max(0.0, start_x) + 1.0, 2), start_z))
        else:
            obstacle = lanes[TILE_NAMES.index(tile)][1:-1]
            # Keep both ends of each riser at the preceding section's elevation.
            points.extend(
                (start_x + (x - obstacle[0][0]), start_z + z) for x, z in obstacle[1:]
            )
            end_x, end_z = points[-1]
            landing = 1.0 if index == len(sequence) - 1 else 0.8
            points.append((round(end_x + landing, 2), end_z))
        sections.append((tile, points[-1][0]))
    finish_x, finish_z = points[-1]
    points.append((finish_x + 12.0, finish_z))
    return points, tuple(sections)


class TerrainCourse:
    def __init__(self, device, *, include_mixed=False, terrain_sequence=None):
        self.offsets = torch.tensor(OFFSETS, device=device)
        self._profiles = profiles()
        self.finish_x, self.x_max = FINISH_X, X_MAX
        self.sections = ()
        if include_mixed:
            sequence = TILE_NAMES if terrain_sequence is None else terrain_sequence
            route, self.sections = mixed_profile(self._profiles, sequence)
            self._profiles = (*self._profiles, route)
            self.finish_x, self.x_max = self.sections[-1][1], route[-1][0]
        grid = np.linspace(
            X_MIN, self.x_max, round((self.x_max - X_MIN) / GRID_SPACING) + 1
        )
        self.heights = torch.as_tensor(
            np.stack(
                [np.interp(grid, *np.asarray(profile).T) for profile in self._profiles]
            ),
            dtype=torch.float32,
            device=device,
        )

    def profile(self, tile):
        return self._profiles[tile]

    def mesh(self):
        vertices, triangles = [], []
        for tile, profile in enumerate(self._profiles):
            start = len(vertices)
            y = tile * LANE_SPACING
            for x, z in profile:
                vertices.extend(((x, y - HALF_WIDTH, z), (x, y + HALF_WIDTH, z)))
            for index in range(len(profile) - 1):
                i = start + 2 * index
                triangles.extend(((i, i + 2, i + 3), (i, i + 3, i + 1)))
        return np.asarray(vertices), np.asarray(triangles)

    def height(self, tile, x):
        # Smooth the riser over one cell for BPTT. Newton uses the triangle mesh.
        coord = ((x - X_MIN) / GRID_SPACING).clamp(0, self.heights.shape[1] - 1)
        lo = coord.long().clamp_max(self.heights.shape[1] - 2)
        return torch.lerp(
            self.heights[tile, lo], self.heights[tile, lo + 1], coord - lo
        )

    def query(self, state, tile):
        p, rotation = state[:2]
        heading = rotation[..., :2, 0]
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (
            p[..., 0, None]
            + heading[..., 0, None] * self.offsets[:, 0]
            - heading[..., 1, None] * self.offsets[:, 1]
        )
        height = self.height(tile[..., None], x)
        support = height[..., :4].mean(-1)
        return height - support[..., None], support

    def sample(self, count, iteration, stairs_start, tall_stairs_start):
        # Keep flat and easy tiles in the mix, and ramp in each pair over 1500 updates.
        low = min(1.0, max(0.0, (iteration - stairs_start + 1) / 1500))
        tall = min(1.0, max(0.0, (iteration - tall_stairs_start + 1) / 1500))
        weights = self.heights.new_tensor((2, 1, 1, 1, 1, 1, low, low, tall, tall))
        return torch.multinomial(weights, count, replacement=True)
