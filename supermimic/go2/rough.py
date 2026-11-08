"""Isaac Lab Go2 Rough tiles and directed traversals for checkpoint evaluation."""

import numpy as np
import torch
import trimesh
import warp as wp
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .terrain import OFFSETS

FAMILIES = {
    "flat": None,
    "slope_up": "hf_pyramid_slope_inv",
    "slope_down": "hf_pyramid_slope",
    "rough": "random_rough",
    "boxes": "boxes",
    "stairs_up": "pyramid_stairs_inv",
    "stairs_down": "pyramid_stairs",
}
RANDOM_FAMILIES = {"boxes", "rough"}
LEVELS = (1, 2, 3, 4, 5)
HEADINGS = tuple(range(0, 360, 45))
TILE_SIZE = 8.0
TILE_SPACING = 8.6
GOAL_EDGE = 3.55
OUTSIDE_EDGE = 3.88
MAX_CROSS_TRACK = 1.0


def make_tile(family, level):
    """Use the pinned Isaac Lab generators and the Go2-specific dimensions."""
    difficulty = (level - 1) / (len(LEVELS) - 1)
    if family == "flat":
        cfg = MeshPlaneTerrainCfg(size=(TILE_SIZE, TILE_SIZE))
        return cfg, difficulty, "level ground"

    base = ROUGH_TERRAINS_CFG
    cfg = base.sub_terrains[FAMILIES[family]].copy()
    cfg.size = (TILE_SIZE, TILE_SIZE)
    if family.startswith("stairs_"):
        cfg.step_height_range = (0.025, 0.12)
        rise = np.interp(difficulty, (0, 1), cfg.step_height_range)
        return cfg, difficulty, f"{rise * 100:.2f} cm rise"
    if family == "boxes":
        cfg.grid_height_range = (0.025, 0.10)
        height = np.interp(difficulty, (0, 1), cfg.grid_height_range)
        return cfg, difficulty, f"{height * 100:.2f} cm height scale"

    cfg.horizontal_scale = base.horizontal_scale
    cfg.vertical_scale = base.vertical_scale
    cfg.slope_threshold = base.slope_threshold
    if family == "rough":
        # The upstream random-height generator ignores difficulty. Vary its ceiling explicitly.
        high = 0.01 * (level + 1)
        cfg.noise_range = (0.01, high)
        cfg.noise_step = 0.01
        return cfg, difficulty, f"1–{level + 1} cm sample heights"
    grade = np.interp(difficulty, (0, 1), cfg.slope_range)
    return cfg, difficulty, f"{grade * 100:.0f}% nominal grade"


@wp.kernel
def _surface_height(
    mesh: wp.uint64,
    xy: wp.array[wp.vec2],
    ray_top: float,
    ray_length: float,
    output: wp.array[float],
):
    i = wp.tid()
    ray = wp.mesh_query_ray(
        mesh, wp.vec3(xy[i][0], xy[i][1], ray_top), wp.vec3(0.0, 0.0, -1.0), ray_length
    )
    output[i] = ray_top - ray.t if ray.result else ray_top - ray_length


class RoughCourse:
    """Shared collision mesh, tile assignments and a terrain-relative height query."""

    def __init__(self, device, families, levels, headings, seeds, *, y_offset=0.0):
        vertices, faces, centers = [], [], []
        self.cases = []
        np_state = np.random.get_state()
        vertex_count = 0
        column = 0
        try:
            for family in families:
                family_seeds = seeds if family in RANDOM_FAMILIES else (None,)
                family_levels = levels[family] if family != "flat" else (1,)
                for seed in family_seeds:
                    for level in family_levels:
                        cfg, difficulty, parameter = make_tile(family, level)
                        if seed is not None:
                            np.random.seed(seed)
                        if family == "boxes":
                            cuda_devices = (
                                [torch.cuda.current_device()]
                                if torch.cuda.is_available()
                                else []
                            )
                            with torch.random.fork_rng(devices=cuda_devices):
                                torch.random.default_generator.manual_seed(seed)
                                if cuda_devices:
                                    torch.cuda.manual_seed(seed)
                                parts, origin = cfg.function(difficulty, cfg)
                        else:
                            parts, origin = cfg.function(difficulty, cfg)
                        tile = trimesh.util.concatenate(parts)
                        offset = np.array(
                            ((level - 1) * TILE_SPACING, y_offset + column * TILE_SPACING, 0.0)
                        )
                        vertices.append(tile.vertices + offset)
                        faces.append(tile.faces + vertex_count)
                        vertex_count += len(tile.vertices)
                        for heading in headings:
                            self.cases.append(
                                {
                                    "family": family,
                                    "level": level,
                                    "difficulty": difficulty,
                                    "parameter": parameter,
                                    "seed": seed,
                                    "heading_deg": heading,
                                    "tile": len(centers),
                                }
                            )
                        centers.append(origin + offset)
                    column += 1
        finally:
            np.random.set_state(np_state)

        self.device = device
        self.vertices = np.concatenate(vertices).astype(np.float32)
        self.faces = np.concatenate(faces).astype(np.int32)
        self.origins = torch.as_tensor(
            np.array(centers), dtype=torch.float32, device=device
        )
        self.trial_tiles = torch.tensor(
            [case["tile"] for case in self.cases], device=device
        )
        self.yaw = torch.tensor(
            [case["heading_deg"] for case in self.cases], device=device
        ).deg2rad()
        self.heading = torch.stack((self.yaw.cos(), self.yaw.sin()), dim=-1)
        self.goal = GOAL_EDGE / self.heading.abs().amax(dim=-1)
        self.offsets = torch.tensor(OFFSETS, device=device)
        self.ray_top = float(self.vertices[:, 2].max() + 2.0)
        self.ray_length = self.ray_top - float(self.vertices[:, 2].min()) + 2.0
        self.surface = wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(self.faces.reshape(-1), dtype=int, device=device),
        )
        self.origins[:, 2] = self._height(self.origins[:, :2])

    def mesh(self):
        return self.vertices, self.faces

    def _height(self, world_xy):
        xy = world_xy.reshape(-1, 2).contiguous()
        result = xy.new_empty(len(xy))
        wp.launch(
            _surface_height,
            dim=len(xy),
            inputs=[
                self.surface.id,
                wp.from_torch(xy, dtype=wp.vec2),
                self.ray_top,
                self.ray_length,
            ],
            outputs=[wp.from_torch(result)],
            device=self.device,
        )
        return result.view(world_xy.shape[:-1])

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
        origin = self.origins[tile]
        points = points.clamp(-TILE_SIZE / 2 + 0.05, TILE_SIZE / 2 - 0.05)
        height = self._height(points + origin[..., None, :2]) - origin[..., 2:3]
        support = height[..., :4].mean(dim=-1)
        return height - support[..., None], support

    def progress(self, position):
        dx, dy = position[:, :2].unbind(dim=-1)
        hx, hy = self.heading.unbind(dim=-1)
        return dx * hx + dy * hy, -dx * hy + dy * hx

    def course_status(self, position):
        along, lateral = self.progress(position)
        reached = (along >= self.goal) & (lateral.abs() <= MAX_CROSS_TRACK)
        outside = position[:, :2].abs().amax(dim=-1) >= OUTSIDE_EDGE
        return reached, outside
