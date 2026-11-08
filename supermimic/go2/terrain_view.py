"""Render colored terrain panels over the active collision lane in Newton GL."""

import math
from bisect import bisect_right
from collections import defaultdict

import numpy as np
import warp as wp

from .terrain import HALF_WIDTH, LANE_SPACING, TILE_NAMES

SURFACE_GRAY = 0.65
EDGE_COLOR = (0.95, 0.95, 0.95)


def show_step_dimensions(viewer, course, tile):
    """Project a height dimension for one riser in each stair or step section."""
    edges = [end for _, end in course.sections]
    risers = defaultdict(list)
    profile = course.profile(tile)
    for (x0, z0), (x1, z1) in zip(profile, profile[1:]):
        if x0 == x1 and z0 != z1:
            risers[bisect_right(edges, x0)].append((x0, min(z0, z1), max(z0, z1)))
    if not risers:
        return

    y = tile * LANE_SPACING - HALF_WIDTH - 0.08
    dimensions = []
    for steps in risers.values():
        x, bottom, top = steps[len(steps) // 2]
        height = f"{100 * (top - bottom):.1f}".removesuffix(".0")
        endpoints = np.array([[x, y, bottom, 1], [x, y, top, 1]])
        dimensions.append((endpoints, f"{height} cm"))

    def draw_dimensions(imgui):
        view = viewer.camera.get_view_matrix().reshape(4, 4).T
        projection = viewer.camera.get_projection_matrix().reshape(4, 4).T
        matrix = projection @ view
        width, height = imgui.get_io().display_size
        draw = imgui.get_foreground_draw_list()
        light = imgui.get_color_u32((0.95, 0.97, 0.99, 1.0))
        dark = imgui.get_color_u32((0.08, 0.12, 0.17, 0.94))
        for world, label in dimensions:
            clip = world @ matrix.T
            if (clip[:, 3] <= 0).any():
                continue
            xy = clip[:, :2] / clip[:, 3:4]
            lower, upper = (
                xy * (width / 2, -height / 2) + (width / 2, height / 2)
            ).tolist()
            x, y = (lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2
            if not (min(380, 0.25 * width) <= x <= width - 82 and 20 <= y <= height - 20):
                continue
            draw.add_line(lower, upper, dark, 4)
            draw.add_line(lower, upper, light, 2)
            for px, py in (lower, upper):
                draw.add_line((px - 6, py), (px + 6, py), dark, 4)
                draw.add_line((px - 6, py), (px + 6, py), light, 2)
            size = imgui.calc_text_size(label)
            draw.add_rect_filled((x + 10, y - 11), (x + 22 + size.x, y + 11), dark, 4)
            draw.add_text((x + 16, y - size.y / 2), light, label)

    viewer.register_ui_callback(draw_dimensions, position="free")


def paint_lane(visualizer, course, tile, device):
    """Submit static display meshes, slightly above the original surface."""
    panels = defaultdict(list)
    center_y = tile * LANE_SPACING
    left, right = center_y - HALF_WIDTH, center_y + HALF_WIDTH
    profile = course.profile(tile)
    section = 0
    band = 0
    for (x0, z0), (x1, z1) in zip(profile, profile[1:]):
        x_mid = (x0 + x1) / 2
        if course.sections:
            while (
                section < len(course.sections) and x_mid >= course.sections[section][1]
            ):
                section += 1
                band = 0
            name = (
                course.sections[section][0] if section < len(course.sections) else "flat"
            )
        else:
            name = TILE_NAMES[tile]
        dx, dz = x1 - x0, z1 - z0
        # Shift along the surface normal, including the vertical risers.
        length = math.hypot(dx, dz)
        x_offset, z_offset = -0.004 * dz / length, 0.004 * dx / length
        count = max(1, math.ceil(dx / 0.55))
        for i in range(count):
            xa, xb = (x0 + dx * j / count + x_offset for j in (i, i + 1))
            za, zb = (z0 + dz * j / count + z_offset for j in (i, i + 1))
            if dx == 0:
                shade = 0.38
            else:
                # Group the short corrugation triangles into visible bands.
                stripe = math.floor(x_mid / 0.28) if name == "bumps" else band
                shade = 1.0 if stripe % 2 == 0 else 0.60
                band += 1
            color = (SURFACE_GRAY * shade,) * 3
            for lo, hi, rgb in (
                (left, left + 0.06, EDGE_COLOR),
                (left + 0.06, right - 0.06, color),
                (right - 0.06, right, EDGE_COLOR),
            ):
                panels[rgb].extend(
                    ((xa, lo, za), (xa, hi, za), (xb, lo, zb), (xb, hi, zb))
                )

    for index, (color, points) in enumerate(panels.items()):
        faces = [i + j for i in range(0, len(points), 4) for j in (0, 2, 3, 0, 3, 1)]
        visualizer.log_mesh(
            f"go2/terrain/{index}",
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(faces, dtype=wp.int32, device=device),
            color=color,
            roughness=0.9,
            backface_culling=False,
        )
