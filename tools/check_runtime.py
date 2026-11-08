"""Check the pinned native GPU runtime without robot assets or training data."""

import argparse
import importlib
import importlib.metadata as metadata
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-physics", action="store_true", help="only check imports and GPU compilation")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The pinned Isaac Lab runtime requires Python 3.12")

    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    for module in ("mujoco", "newton", "warp", "pxr.Usd", "isaacsim.asset.importer.mjcf"):
        importlib.import_module(module)

    for package in ("torch", "warp-lang", "mujoco", "mujoco-warp", "newton", "isaaclab", "usd-exchange"):
        print(f"{package}: {metadata.version(package)}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the NVIDIA GPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    def objective(x):
        return x.sin().square().mean()

    x = torch.randn(256, device="cuda", requires_grad=True)
    torch.compile(objective)(x).backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("torch.compile GPU backward produced non-finite gradients")
    print("torch.compile GPU forward/backward: OK", flush=True)
    if args.skip_physics:
        return

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass, instantiate

    @configclass
    class RuntimeSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.1,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

    cfg = sim_utils.SimulationCfg(
        dt=0.02, device="cuda:0",
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=64, nconmax=32, use_mujoco_contacts=True, integrator="implicitfast"),
            num_substeps=4,
        ),
    )
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene = instantiate(RuntimeSceneCfg(num_envs=1, env_spacing=0.0, replicate_physics=True))
        sim.reset()
        initial_z = float(scene["ball"].data.root_link_pose_w.torch[0, 2])
        for _ in range(100):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(0.02)
        final_z = float(scene["ball"].data.root_link_pose_w.torch[0, 2])
        if not 0.05 < final_z < 0.2 or initial_z < 0.9:
            raise RuntimeError(f"Ground contact check failed: z {initial_z} -> {final_z}")
        print(f"Isaac Lab / Newton / MuJoCo-Warp GPU contact: OK (z {initial_z:.3f} -> {final_z:.3f})", flush=True)


if __name__ == "__main__":
    main()
