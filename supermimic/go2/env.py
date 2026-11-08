"""Batched Go2 on flat ground or a terrain course in Newton / MuJoCo-Warp."""

import math

import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from newton import ModelFlags

from supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp

from .commands import advance_commands, sample_commands
from .config import CONTROL_DECIMATION, NUM_J, PHYSICS_DT, XI_DIM, CommandCfg, TaskCfg
from .mdp import VelocityMDP, flatten_state, route_reward
from .terrain import (
    EVAL_TILE_NAMES,
    FOOT_NAMES,
    HALF_WIDTH,
    LANE_SPACING,
    MIXED_TILE,
    X_MIN,
    TerrainCourse,
)


def make_sim(device, *, terrain_course=False):
    return sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        use_newton_actuators=True,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=256 if terrain_course else 128,
                nconmax=96 if terrain_course else 64,
                cone="pyramidal",
                integrator="implicitfast",
                # Use Newton's concave mesh contacts for the course risers.
                use_mujoco_contacts=not terrain_course,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
            if terrain_course
            else None,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)
            if terrain_course
            else NewtonShapeCfg(),
            num_substeps=2,
        ),
    )


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    feet = None


class VelocitySim:
    def __init__(
        self,
        sim_cfg,
        num_envs,
        task: TaskCfg,
        commands: CommandCfg,
        *,
        eval_mode=False,
        asset=None,
        terrain_tile="step_up",
        terrain_sequence=None,
        rough_course=None,
        rough_curriculum=None,
        stairs_start=4000,
        tall_stairs_start=9000,
    ):
        self.device, self.N = sim_cfg.device, num_envs
        self.task, self.command_cfg, self.eval_mode = task, commands, eval_mode
        if rough_course is not None and (not eval_mode or not task.terrain_course):
            raise ValueError("Go2 Rough requires evaluation with a terrain checkpoint")
        if rough_curriculum is not None and (eval_mode or not task.terrain_course):
            raise ValueError("The rough curriculum requires terrain training")
        self.rough_course = rough_course
        self.rough_curriculum = rough_curriculum
        self.command_scale = (
            1.0 if eval_mode or task.terrain_course else commands.scale(0)
        )
        self.terrain = rough_course if eval_mode else rough_curriculum
        if self.terrain is None and task.terrain_course:
            self.terrain = TerrainCourse(
                self.device,
                include_mixed=eval_mode and terrain_tile == MIXED_TILE,
                terrain_sequence=terrain_sequence,
            )
        self.terrain_iteration = 0
        self.stairs_start, self.tall_stairs_start = stairs_start, tall_stairs_start
        self.eval_tile = (
            EVAL_TILE_NAMES.index(terrain_tile) if self.terrain is not None else 0
        )
        self.sim = SimulationContext(sim_cfg)
        scene_cfg = Go2SceneCfg(
            num_envs=num_envs, env_spacing=2.5, replicate_physics=True
        )
        if self.terrain is not None:
            vertices, faces = self.terrain.mesh()
            scene_cfg.ground = AssetBaseCfg(
                prim_path="/World/Go2Course",
                spawn=sim_utils.MeshFileCfg(
                    mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(
                        vertices=vertices, faces=faces
                    ),
                    collision_props=[
                        sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
                    ],
                    mesh_collision_props=[
                        sim_utils.UsdPhysicsMeshCollisionCfg(
                            mesh_approximation_name="none"
                        )
                    ],
                ),
            )
            scene_cfg.feet = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/(FL|FR|RL|RR)_foot",
                track_friction_forces=False,
            )
        spawn_options = {"activate_contact_sensors": self.terrain is not None}
        if asset:
            spawn_options["usd_path"] = asset
        scene_cfg.robot = replace(
            scene_cfg.robot, spawn=replace(scene_cfg.robot.spawn, **spawn_options)
        )
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        self.sim.backend_manager.set_decimation(CONTROL_DECIMATION)
        self.steps_per_call = (
            CONTROL_DECIMATION if self.sim.backend_manager.handles_decimation() else 1
        )
        self.robot = self.scene["robot"]
        self.origins = self.scene.env_origins.clone()
        self.contacts = None
        self.route_heading = None
        if self.rough_curriculum is not None:
            self.rough_yaw = torch.zeros(num_envs, device=self.device)
            if self.rough_curriculum.route_start is not None:
                self.route_heading = torch.zeros(num_envs, 2, device=self.device)
        if self.terrain is not None:
            self.terrain_id = torch.zeros(
                num_envs, dtype=torch.long, device=self.device
            )
            self.feet = self.scene["feet"]
            names = [name.rsplit("/", 1)[-1] for name in self.feet.sensor_names]
            self.foot_order = torch.tensor(
                [names.index(name) for name in FOOT_NAMES], device=self.device
            )
            self.contacts = torch.zeros(num_envs, len(FOOT_NAMES), device=self.device)
        self.joint_names = list(self.robot.joint_names)
        if len(self.joint_names) != NUM_J:
            raise ValueError(f"Expected 12 Go2 joints, got {self.joint_names}")
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        limits = self.robot.data.joint_pos_limits.torch[0]
        self.lower, self.upper = limits[:, 0].clone(), limits[:, 1].clone()
        self.mdp = VelocityMDP(self.default_q, self.lower, self.upper, task)
        self.xi = torch.zeros(num_envs, XI_DIM, device=self.device)
        self.xi[:, 0] = (
            1.0
            if eval_mode
            else torch.empty(num_envs, device=self.device).uniform_(0.3, 1.2)
        )
        self._set_friction()
        self.a_prev = torch.zeros(num_envs, NUM_J, device=self.device)
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.command = torch.zeros(num_envs, 3, device=self.device)
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(num_envs, device=self.device)
        self.reset(torch.arange(num_envs, device=self.device))

    def _set_friction(self):
        manager = self.sim.backend_manager
        model = manager.get_model()
        mu = wp.to_torch(model.shape_material_mu)
        mu[wp.to_torch(model.shape_body) < 0] = 1.0 if self.eval_mode else 0.3
        view = wp.to_torch(
            self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0]
        )
        view[:] = self.xi[:, :1]
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def state(self):
        data = self.robot.data
        pose, velocity = (
            data.root_link_pose_w.torch.clone(),
            data.root_link_vel_w.torch.clone(),
        )
        # Isaac Lab 3 stores xyzw; the shared rotation utilities take wxyz.
        quaternion = torch.cat((pose[:, 6:7], pose[:, 3:6]), -1)
        return (
            pose[:, :3] - self.origins,
            quat_to_mat(quaternion),
            velocity[:, :3],
            velocity[:, 3:],
            data.joint_pos.torch.clone(),
            data.joint_vel.torch.clone(),
        )

    def observe(self, state=None):
        state = self.state() if state is None else state
        return self.mdp.obs(state, self.a_prev, self.xi, self.contacts)

    def reset(self, ids):
        if len(ids) == 0:
            return
        count = len(ids)
        default = torch.cat(
            (
                self.robot.data.default_root_pose.torch[ids],
                self.robot.data.default_root_vel.torch[ids],
            ),
            -1,
        )
        if self.rough_course is not None:
            tiles = self.rough_course.trial_tiles[ids]
            self.terrain_id[ids] = tiles
            self.origins[ids] = self.rough_course.origins[tiles]
            self.contacts[ids] = 0
            half_yaw = self.rough_course.yaw[ids] / 2
            default[:, 3:5] = 0
            default[:, 5] = half_yaw.sin()
            default[:, 6] = half_yaw.cos()
        elif self.terrain is not None:
            tiles = (
                torch.full_like(ids, self.eval_tile)
                if self.eval_mode
                else self.terrain.sample(
                    count,
                    self.terrain_iteration,
                    self.stairs_start,
                    self.tall_stairs_start,
                )
            )
            self.terrain_id[ids] = tiles
            if self.rough_curriculum is None:
                self.origins[ids] = 0
                self.origins[ids, 1] = tiles * LANE_SPACING
            else:
                self.origins[ids] = self.rough_curriculum.origins[tiles]
            self.contacts[ids] = 0
        default[:, :3] += self.origins[ids]
        q = self.default_q.expand(count, -1).clone()
        l5_focus = 0.0
        l5_perturb = 0.0
        if not self.eval_mode:
            rotation = torch.zeros(count, 3, device=self.device)
            rotation[:, :2].uniform_(-0.05, 0.05)
            rotation[:, 2].uniform_(-math.pi, math.pi)
            if self.terrain is not None:
                rotation[:, 2] = torch.where(
                    tiles == 0, rotation[:, 2], rotation[:, 2] * (0.12 / math.pi)
                )
                spawn = torch.rand(count, device=self.device)
                default[:, 0] += spawn * 0.16
                if self.rough_curriculum is not None:
                    l5_focus = self.rough_curriculum.l5_focus(self.terrain_iteration)
                    l5_perturb = max(
                        l5_focus,
                        self.rough_curriculum.l5_coverage(self.terrain_iteration),
                    )
                    if l5_perturb:
                        focused = l5_perturb * self.rough_curriculum.l5_stairs(tiles)
                        rotation[:, 2] *= 1 + focused * (math.radians(12) / 0.12 - 1)
                    yaw = self.rough_curriculum.initial_yaw(tiles, rotation[:, 2])
                    rotation[:, 2] = yaw
                    self.rough_yaw[ids] = yaw
                    if l5_perturb or self.route_heading is not None:
                        heading = torch.stack((yaw.cos(), yaw.sin()), -1)
                    if l5_perturb:
                        # Move the spawn along its route to vary the approach to the first stair.
                        default[:, 0] -= focused * spawn * 0.16
                        default[:, :2] += (focused * (spawn - 0.5) * 0.30)[:, None] * heading
                    if self.route_heading is not None:
                        guided = self.rough_curriculum.guided_stairs(tiles, self.terrain_iteration)
                        self.route_heading[ids] = heading * guided[:, None]
            quaternion = mat_to_quat(so3_exp(rotation))
            default[:, 3:7] = torch.cat((quaternion[:, 1:], quaternion[:, :1]), -1)
            default[:, 7:13].uniform_(-0.05, 0.05)
            q.add_(torch.empty_like(q).uniform_(-0.05, 0.05))
            self.xi[ids, 1:] = torch.empty_like(q).uniform_(-0.01, 0.01)
        q = q.clamp(self.lower, self.upper)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=default[:, :7], env_ids=ids
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=default[:, 7:13], env_ids=ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(q), env_ids=ids
        )
        self.robot.set_joint_position_target_index(target=q, env_ids=ids)
        self.a_prev[ids] = 0
        self.ep_step[ids] = 0
        self.episode_return[ids] = 0
        self.command[ids], self.remaining[ids] = sample_commands(
            count, self.device, self.command_cfg, self.command_scale
        )
        if self.terrain is not None and not self.eval_mode:
            forward = torch.zeros(count, 3, device=self.device)
            forward[:, 0].uniform_(0.28, 0.50)
            if l5_focus and self.route_heading is not None:
                faster = guided & (torch.rand(count, device=self.device) < 0.5 * l5_focus)
                upper_speed = torch.empty_like(forward[:, 0]).uniform_(0.40, 0.50)
                forward[:, 0] = torch.where(faster, upper_speed, forward[:, 0])
            self.command[ids] = torch.where(
                (tiles != 0)[:, None], forward, self.command[ids]
            )
            self.remaining[ids] = torch.where(
                tiles != 0, self.command_cfg.max_steps, self.remaining[ids]
            )
        self.scene.reset(ids)
        self.sim.forward()
        self.scene.update(0.0)

    def step(self, action, observation):
        # Store the action actually executed, before any terminal-state reset.
        action = action.clamp(-self.task.action_clip, self.task.action_clip)
        state = self.state()
        previous = self.a_prev.clone()
        command, remaining = self.command.clone(), self.remaining.clone()
        ep_step, xi = self.ep_step.clone(), self.xi.clone()
        target = self.default_q + xi[:, 1:] + self.task.action_scale * action
        self.robot.set_joint_position_target_index(target=target)
        # DC motor feedback must run at 200 Hz. Increasing solver substeps alone
        # would still hold the explicit PD torque for an entire 20 ms control tick.
        for _ in range(CONTROL_DECIMATION // self.steps_per_call):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT * self.steps_per_call)
        following = self.state()
        support = 0.0
        fixed = None
        curriculum = self.rough_curriculum
        rough = self.terrain_id >= curriculum.first_tile if curriculum is not None else None
        if self.terrain is not None:
            _, support = self.terrain.query(following, self.terrain_id)
            forces = self.feet.data.net_forces_w.torch[:, self.foot_order]
            contacts1 = forces.square().sum(-1) > 25.0  # 5 N contact threshold.
            self.contacts.copy_(contacts1)
            fixed = self.terrain_id != 0
        reward, metrics = self.mdp.reward(
            following, command, action, previous, terms=True, support=support
        )
        if self.route_heading is not None:
            reward = reward + route_reward(following, command, self.route_heading)
        terminal = self.mdp.termination(following, support)
        done = terminal | (ep_step + 1 >= self.task.episode_steps)
        if self.rough_course is not None:
            reached, outside = self.rough_course.course_status(following[0])
            reached = reached & ~terminal & ~outside
            done = done | reached | outside
        elif self.terrain is not None:
            position = following[0]
            outside = (
                (position[:, 0] < X_MIN + 0.4)
                | (position[:, 0] > self.terrain.x_max - 0.4)
                | (position[:, 1].abs() > HALF_WIDTH - 0.4)
            )
            reached = (
                fixed & (position[:, 0] >= self.terrain.finish_x) & ~terminal & ~outside
            )
            if curriculum is not None:
                directed, beyond = curriculum.course_status(position, self.rough_yaw)
                outside = torch.where(rough, beyond, outside)
                reached = torch.where(rough, directed & ~terminal & ~outside, reached)
            done = done | reached | outside
        transition = {
            "s": flatten_state(state),
            "s1": flatten_state(following),
            "a": action.clone(),
            "a_prev": previous,
            "xi": xi,
            "obs": observation.clone(),
            "command": command,
            "remaining": remaining,
            "ep_step": ep_step,
            "r": reward,
            "term": terminal,
            "done": done,
        }
        if self.terrain is not None:
            transition.update(
                terrain_id=self.terrain_id.clone(),
                contacts1=contacts1,
                course_reached=reached,
            )
            metrics["contact_fraction"] = contacts1.float().mean()
            if self.rough_course is None:
                stairs = self.terrain_id >= 6
                if curriculum is not None:
                    family = curriculum.family[(self.terrain_id - curriculum.first_tile).clamp_min(0)]
                    stairs = torch.where(rough, family < 2, stairs)
                    metrics["rough_fraction"] = rough.float().mean()
                metrics["stairs_fraction"] = stairs.float().mean()
            else:
                transition["course_outside"] = outside
        if self.route_heading is not None:
            transition["route_heading"] = self.route_heading.clone()
        self.a_prev.copy_(action)
        self.ep_step += 1
        self.episode_return += reward
        if not self.eval_mode:
            self.command, self.remaining = advance_commands(
                command, remaining, self.command_cfg, self.command_scale, fixed
            )
            transition["command1"] = self.command.clone()
            transition["remaining1"] = self.remaining.clone()
            ids = done.nonzero().flatten()
            if len(ids):
                metrics["episode_return"] = self.episode_return[ids].clone()
                metrics["episode_length"] = self.ep_step[ids].clone()
                if self.terrain is not None:
                    course_ids = ids[fixed[ids]]
                    if len(course_ids):
                        metrics["course_success"] = reached[course_ids].float()
                    if curriculum is not None:
                        legacy_ids = course_ids[~rough[course_ids]]
                        if len(legacy_ids):
                            metrics["rough_legacy_success"] = reached[legacy_ids].float()
                        for index, name in enumerate(curriculum.rate_keys):
                            matching = ids[rough[ids] & (family[ids] == index)]
                            if len(matching):
                                metrics[name] = reached[matching].float()
                        if curriculum.level_rate_keys:
                            rough_ids = ids[rough[ids]]
                            levels = curriculum.level[self.terrain_id[rough_ids] - curriculum.first_tile]
                            for (index, level), name in curriculum.level_rate_keys.items():
                                matching = rough_ids[(family[rough_ids] == index) & (levels == level)]
                                if len(matching):
                                    metrics[name] = reached[matching].float()
                self.reset(ids)
        return transition, metrics
