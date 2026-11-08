"""Mixed legacy lanes and differentiable 2D Go2 Rough tiles for fine-tuning."""

import math

import numpy as np
import torch

from .rough import GOAL_EDGE, MAX_CROSS_TRACK, OUTSIDE_EDGE, TILE_SIZE, RoughCourse
from .terrain import GRID_SPACING, LANE_SPACING, TILE_NAMES, X_MAX, TerrainCourse, profiles

ROUGH_FAMILIES = ("stairs_up", "stairs_down", "boxes")
ROUGH_LEVELS = {"stairs_up": (2, 3, 4), "stairs_down": (2, 3, 4), "boxes": (2, 3)}
ROUGH_FIRST = len(TILE_NAMES)
ROUGH_RATE_KEYS = tuple(f"rough_{name}_success" for name in ROUGH_FAMILIES)
STAIR_LEVEL_RATES = {
    (index, level): f"rough_{family}_l{level}_success"
    for index, family in enumerate(ROUGH_FAMILIES[:2])
    for level in (4, 5)
}
HEIGHT_X_MIN = -TILE_SIZE / 2
L5_FOCUS_RAMP = 4000


class RoughCurriculum(TerrainCourse):
    first_tile = ROUGH_FIRST
    rate_keys = ROUGH_RATE_KEYS

    def __init__(
        self, device, start, ramp, *, seed, route_start=None,
        l5_focus_start=None, l5_coverage_start=None,
    ):
        super().__init__(device)
        self.start, self.ramp = start, ramp
        self.route_start = route_start
        self.l5_focus_start = l5_focus_start
        self.l5_coverage_start = l5_coverage_start
        self.level_rate_keys = STAIR_LEVEL_RATES if route_start is not None else {}
        levels = ROUGH_LEVELS
        if route_start is not None:
            levels = {
                family: (*family_levels, 5) if family in ROUGH_FAMILIES[:2] else family_levels
                for family, family_levels in ROUGH_LEVELS.items()
            }
        rough = RoughCourse(
            device,
            ROUGH_FAMILIES,
            levels,
            (0,),
            tuple(seed + index for index in range(4)),
            y_offset=(ROUGH_FIRST + 1) * LANE_SPACING,
        )
        old_vertices, old_faces = super().mesh()
        new_vertices, new_faces = rough.mesh()
        self.vertices = np.concatenate((old_vertices, new_vertices))
        self.faces = np.concatenate((old_faces, new_faces + len(old_vertices)))

        legacy_origins = torch.zeros(ROUGH_FIRST, 3, device=device)
        legacy_origins[:, 1] = torch.arange(ROUGH_FIRST, device=device) * LANE_SPACING
        self.origins = torch.cat((legacy_origins, rough.origins), dim=0)
        self.family = torch.tensor(
            [ROUGH_FAMILIES.index(case["family"]) for case in rough.cases], device=device
        )
        self.level = torch.tensor([case["level"] for case in rough.cases], device=device)
        if route_start is not None:
            tile_groups = [f"legacy_{name}" for name in TILE_NAMES] + [
                f"rough_{case['family']}_l{case['level']}" for case in rough.cases
            ]
            self.sample_groups = tuple(dict.fromkeys(tile_groups))
            self.sample_group_ids = torch.tensor(
                [self.sample_groups.index(group) for group in tile_groups], device=device
            )

        # Bake ray hits once. The Torch lookup remains differentiable in x and y
        # during the imagined rollout, including on tiles that vary across both axes.
        width = round(TILE_SIZE / GRID_SPACING) + 1
        grid = torch.linspace(-TILE_SIZE / 2, TILE_SIZE / 2, width, device=device)
        xy = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1)
        xy = xy.clamp(-TILE_SIZE / 2 + 0.005, TILE_SIZE / 2 - 0.005)
        grids = torch.stack([rough._height(xy + o[:2]) - o[2] for o in rough.origins])
        num_x = round((X_MAX - HEIGHT_X_MIN) / GRID_SPACING) + 1
        grid_x = np.linspace(HEIGHT_X_MIN, X_MAX, num_x)
        legacy = torch.as_tensor(
            np.stack([np.interp(grid_x, *np.asarray(p).T) for p in profiles()]),
            device=device,
            dtype=torch.float32,
        )
        legacy = legacy[..., None].expand(-1, -1, width)
        extra_x = grids[:, -1:].expand(-1, num_x - width, -1)
        self.surface_grid = torch.cat((legacy, torch.cat((grids, extra_x), dim=1)), dim=0)

    def mesh(self):
        return self.vertices, self.faces

    def l5_focus(self, iteration):
        if self.l5_focus_start is None:
            return 0.0
        return min(1.0, max(0.0, (iteration - self.l5_focus_start) / L5_FOCUS_RAMP))

    def l5_stairs(self, tiles):
        index = (tiles - ROUGH_FIRST).clamp_min(0)
        return (tiles >= ROUGH_FIRST) & (self.family[index] < 2) & (self.level[index] == 5)

    def l5_coverage(self, iteration):
        if self.l5_coverage_start is None:
            return 0.0
        return min(1.0, max(0.0, (iteration - self.l5_coverage_start) / 2000))

    def sample(self, count, iteration, stairs_start, tall_stairs_start):
        legacy = super().sample(count, iteration, stairs_start, tall_stairs_start)
        if iteration < self.start:
            return legacy
        age = iteration - self.start
        ramp = lambda elapsed, span: min(1.0, max(0.0, elapsed / span))
        rough_fraction = 0.15 + 0.50 * ramp(age, self.ramp)
        level3 = 0.15 + 0.85 * ramp(age, self.ramp * 0.5)
        level4 = ramp(age - self.ramp * 0.5, self.ramp * 0.5)
        weights = torch.where(self.level == 2, 1.0, torch.where(self.level == 3, level3, level4))
        if self.route_start is not None:
            tall = ramp(iteration - self.route_start - 2000, 3000)
            weights = torch.where(self.level == 5, tall, weights)
        focus = self.l5_focus(iteration)
        if focus:
            # Preserve the L4 share and shift equal mass from L2 and L3 to L5.
            stair_weights = torch.where(
                self.level == 5, 1 + focus, torch.where(self.level == 4, 1.0, 1 - 0.5 * focus)
            )
            weights = torch.where(self.family < 2, stair_weights, weights)
        # Give each family the same share regardless of difficulty levels or mesh seeds.
        totals = torch.zeros(len(ROUGH_FAMILIES), device=weights.device)
        totals.scatter_add_(0, self.family, weights)
        rough = torch.multinomial(weights / totals[self.family], count, replacement=True)
        use_rough = torch.rand(count, device=weights.device) < rough_fraction
        return torch.where(use_rough, rough + ROUGH_FIRST, legacy)

    def guided_stairs(self, tiles, iteration):
        if self.route_start is None or iteration < self.route_start:
            return torch.zeros_like(tiles, dtype=torch.bool)
        family = self.family[(tiles - ROUGH_FIRST).clamp_min(0)]
        return (tiles >= ROUGH_FIRST) & (family < 2)

    def initial_yaw(self, tiles, legacy_yaw):
        sector = torch.randint(4, tiles.shape, device=tiles.device) * (math.pi / 2)
        diagonal = torch.rand(tiles.shape, device=tiles.device) < 0.75
        yaw = sector + diagonal * (math.pi / 4) + legacy_yaw
        yaw = (yaw + math.pi).remainder(2 * math.pi) - math.pi
        return torch.where(tiles >= ROUGH_FIRST, yaw, legacy_yaw)

    def query(self, state, tile):
        position, rotation = state[:2]
        front = rotation[..., :2, 0]
        front = front / front.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        side = torch.stack((-front[..., 1], front[..., 0]), dim=-1)
        points = (
            position[..., None, :2]
            + front[..., None, :] * self.offsets[:, :1]
            + side[..., None, :] * self.offsets[:, 1:]
        )
        heights = self.surface_grid
        x = ((points[..., 0] - HEIGHT_X_MIN) / GRID_SPACING).clamp(0, heights.shape[1] - 1)
        y = ((points[..., 1] + TILE_SIZE / 2) / GRID_SPACING).clamp(0, heights.shape[2] - 1)
        # Legacy lanes clamp to the original x range.
        x = torch.where(tile[..., None] < ROUGH_FIRST, x.clamp_min(1.0 / GRID_SPACING), x)
        xi = x.long().clamp_max(heights.shape[1] - 2)
        yi = y.long().clamp_max(heights.shape[2] - 2)
        tile = tile[..., None]
        lower = torch.lerp(heights[tile, xi, yi], heights[tile, xi + 1, yi], x - xi)
        upper = torch.lerp(heights[tile, xi, yi + 1], heights[tile, xi + 1, yi + 1], x - xi)
        height = torch.lerp(lower, upper, y - yi)
        support = height[..., :4].mean(dim=-1)
        return height - support[..., None], support

    @staticmethod
    def course_status(position, yaw):
        hx, hy = yaw.cos(), yaw.sin()
        x, y = position[:, :2].unbind(dim=-1)
        along = x * hx + y * hy
        lateral = -x * hy + y * hx
        goal = GOAL_EDGE / torch.maximum(hx.abs(), hy.abs())
        reached = (along >= goal) & (lateral.abs() <= MAX_CROSS_TRACK)
        outside = position[:, :2].abs().amax(dim=-1) >= OUTSIDE_EDGE
        return reached, outside
