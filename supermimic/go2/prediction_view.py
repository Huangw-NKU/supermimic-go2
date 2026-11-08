"""Roll out the trained Go2 world model and draw its predicted pose in Newton GL."""

import newton
import torch
import warp as wp
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab_visualizers.newton.newton_visualization_markers import (
    NewtonVisualizationMarkers,
)

from supermimic.rotations import mat_to_quat

from .mdp import normalize_xi
from .networks import WorldModel
from .terrain import TERRAIN_DIM


@wp.kernel
def _set_free_joint(
    root_pose: wp.array[wp.transform],
    parent_frame: wp.array[wp.transform],
    child_frame: wp.array[wp.transform],
    joint: int,
    start: int,
    joint_q: wp.array[float],
):
    relative = (
        wp.transform_inverse(parent_frame[joint]) * root_pose[0] * child_frame[joint]
    )
    for i in range(7):
        joint_q[start + i] = relative[i]


@wp.kernel
def _shape_poses(
    shape_ids: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    poses: wp.array[wp.transform],
):
    i = wp.tid()
    shape = shape_ids[i]
    poses[i] = body_q[shape_body[shape]] * shape_transform[shape]


class PredictionGhost(NewtonVisualizationMarkers):
    """Use a private FK state and instance the robot's visual meshes."""

    def __init__(self, env, net_cfg, weights, steps):
        super().__init__(
            VisualizationMarkersCfg(prim_path="/Go2_world_prediction", markers={})
        )
        self.env, self.steps = env, steps
        terrain_dim = TERRAIN_DIM if env.terrain is not None else 0
        self.world_model = WorldModel(net_cfg, terrain_dim)
        self.world_model.to(env.device).eval()
        self.world_model.load_state_dict(weights)
        self.model = env.sim.backend_manager.get_model()
        self.state = self.model.state()
        self.joint_q = wp.to_torch(self.state.joint_q)
        joint_types = self.model.joint_type.numpy()
        roots = [i for i, kind in enumerate(joint_types) if kind == newton.JointType.FREE]
        if len(roots) != 1:
            raise ValueError("The Go2 prediction overlay requires one floating root joint")
        self.root_joint = roots[0]
        starts = self.model.joint_q_start.numpy()
        self.root_start = int(starts[self.root_joint])
        joint_starts = {
            name.rsplit("/", 1)[-1]: int(start)
            for name, start in zip(self.model.joint_label, starts)
        }
        self.leg_coords = torch.tensor(
            [joint_starts[name] for name in env.joint_names], device=env.device
        )

        parents, flags = self.model.shape_body.numpy(), self.model.shape_flags.numpy()
        shapes = [
            i
            for i, (parent, flag, source) in enumerate(
                zip(parents, flags, self.model.shape_source)
            )
            if parent >= 0
            and flag & int(newton.ShapeFlags.VISIBLE)
            and isinstance(source, newton.Mesh)
        ]
        if not shapes:
            raise ValueError("The Go2 prediction overlay could not find any visual link meshes")
        self.meshes = [self.model.shape_source[i] for i in shapes]
        self.shape_ids = wp.array(shapes, dtype=int, device=env.device)
        self.transforms = wp.empty(len(shapes), dtype=wp.transform, device=env.device)
        self.scales = wp.array(
            self.model.shape_scale.numpy()[shapes], dtype=wp.vec3, device=env.device
        )
        self.color = wp.array([(0.10, 0.81, 1.0)], dtype=wp.vec3, device=env.device)
        self.material = wp.array([(0.75, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=env.device)
        self.opacity = wp.array([0.54], dtype=float, device=env.device)
        self.ready = False
        self.registered = False

    def forecast(self, policy, state, history):
        """Keep the current command and feed predicted contact bits back to the actor."""
        env = self.env
        xi_normalized = normalize_xi(env.xi)
        for _ in range(self.steps):
            action = policy(history, env.command)
            ground, support = (
                (None, 0.0)
                if env.terrain is None
                else env.terrain.query(state, env.terrain_id)
            )
            state, logits = self.world_model.predict(
                state, action, xi_normalized, ground, support
            )
            contacts = None if env.terrain is None else (logits > 0).to(logits.dtype)
            observation = env.mdp.obs(state, action, env.xi, contacts)
            history = torch.cat((history[:, 1:], observation[:, None]), dim=1)

        position, rotation, _, _, joints, _ = state
        quaternion = mat_to_quat(rotation)
        root = torch.cat(
            (position + env.origins, quaternion[:, 1:], quaternion[:, :1]), dim=-1
        )
        self.ready = torch.isfinite(root).all().item() and torch.isfinite(joints).all().item()
        if not self.ready:
            return
        self.joint_q[self.leg_coords] = joints[0]
        wp.launch(
            _set_free_joint,
            dim=1,
            inputs=[
                wp.from_torch(root, dtype=wp.transform),
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.root_joint,
                self.root_start,
            ],
            outputs=[self.state.joint_q],
            device=env.device,
        )
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        wp.launch(
            _shape_poses,
            dim=len(self.meshes),
            inputs=[
                self.shape_ids,
                self.model.shape_body,
                self.model.shape_transform,
                self.state.body_q,
            ],
            outputs=[self.transforms],
            device=env.device,
        )

    def render(self, viewer, visible_env_ids, num_envs):
        show = self.visible and self.ready
        if not show and not self.registered:
            return
        for i, mesh in enumerate(self.meshes):
            name = f"/go2_prediction/{i}"
            prototype = f"{name}/mesh"
            if not self.registered:
                viewer.log_mesh(
                    prototype,
                    wp.array(mesh.vertices, dtype=wp.vec3, device=viewer.device),
                    wp.array(mesh.indices, dtype=int, device=viewer.device),
                    normals=wp.array(mesh.normals, dtype=wp.vec3, device=viewer.device)
                    if mesh.normals is not None
                    else None,
                    hidden=True,
                )
            viewer.log_instances(
                name,
                prototype,
                self.transforms[i:i+1] if show else None,
                self.scales[i:i+1] if show else None,
                self.color,
                self.material,
                hidden=not show,
                opacities=self.opacity,
            )
        self.registered = True
