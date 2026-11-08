# SuperMimic Go2

Go2 velocity control through a learned differentiable world model, with
flat-ground training, proprioceptive terrain locomotion, rough-terrain
evaluation, and keyboard playback.

This is an unofficial quadruped extension inspired by
[SuperMimic](https://arxiv.org/abs/2610.28480), extracted from the Go2 extension in
[supermimic_vibed](https://github.com/Huangw-NKU/supermimic_vibed).
It does not use reference motions or require a motion dataset.

## Features

- Jointly trained velocity-command encoder and action decoder.
- Learned dynamics model and differentiable 16-step imagined rollouts.
- Ten-frame proprioceptive history: 42 features per frame on flat ground,
  or 46 with four binary foot-contact inputs on terrain.
- Privileged nine-point terrain heights for the world model and critic;
  the deployed policy does not receive terrain heights, camera images, or lidar.
- Flat ground, slopes, uneven ground, single steps, stairs, and random blocks.
- Multi-terrain evaluation across difficulty levels, headings, and geometry seeds.
- Newton GL keyboard playback, foot-contact indicators, and optional predicted poses.

Simulation uses **Isaac Lab + Newton with the MuJoCo Warp solver**, without
launching Isaac Sim / Kit. The policy runs at 50 Hz, motor feedback at 200 Hz,
and physics substeps at 400 Hz. Real-robot deployment has not been validated.

## Installation

Requirements: Linux, an NVIDIA GPU, a driver compatible with the pinned CUDA 13
runtime, Conda, Git, and GCC/G++.

```bash
./tools/setup_native.sh
conda activate supermimic-go2
python tools/check_go2.py --visualizer none
```

The installer creates a dedicated Conda environment and pins Isaac Lab to
[`46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`](https://github.com/isaac-sim/IsaacLab/commit/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed).
Newton, MuJoCo Warp, PyTorch, and USD importers are installed from that checkout's
frozen dependency export. Set `SUPERMIMIC_ENV` to change the environment name or
`SUPERMIMIC_ISAACLAB_DIR` to select another checkout at the same pinned revision.

If the pinned simulation runtime is already installed:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Installing this package alone does not install the complete simulation runtime.
Go2 assets are resolved through Isaac Lab and may require a network connection
on first use. For offline assets, pass `--asset /path/to/go2.usd` and preserve
its relative USD dependencies, or set `ISAACSIM_ASSET_ROOT` to a complete local
Isaac asset root. Robot assets are not included in this repository.

## Training

Start with the smaller default flat-ground configuration:

```bash
./run_go2_train.sh go2_flat --iterations 32000 --visualizer none
```

The four presets in `configs/` provide the historical training stages:

```bash
# Optional larger flat-ground configuration, ending at iteration 32,000.
python tools/train_preset.py flat_32k go2_flat

# Initialize the terrain policy from a flat-ground checkpoint.
# The new stage starts from iteration zero and trains to 20,000.
python tools/train_preset.py proprio_course_20k go2_course \
  --init_flat logs/go2_flat/ckpt_032000.pt

# Resume with two-dimensional rough terrain, from 20,000 to 32,000.
python tools/train_preset.py rough_32k go2_rough \
  --resume logs/go2_course/ckpt_020000.pt

# Resume with symmetric stair sampling and L5 stairs, up to 51,000.
python tools/train_preset.py balanced_51k go2_balanced \
  --resume logs/go2_rough/ckpt_032000.pt
```

Use either the default flat-ground command or the larger flat preset in a fresh
run directory. The historical `flat_32k` and `balanced_51k` presets use 8,192 and
16,384 environments, respectively. Adjust `--num_envs`, `--buffer_steps`,
`--wm_batch`, and `--pi_batch` to fit the GPU. For example:

```bash
python tools/train_preset.py flat_32k go2_flat_small \
  --num_envs 512 --buffer_steps 500 --wm_batch 512 --pi_batch 512

# Inspect a command without launching training.
python tools/train_preset.py balanced_51k go2_balanced \
  --resume logs/go2_rough/ckpt_032000.pt --dry-run
```

Checkpoints are saved every **1,000 iterations**. `--iterations` is the absolute
target for the current stage. `--init_flat` starts new terrain optimizers;
`--resume` restores networks, normalizers, optimizers, and random-number states.
Physics state and replay data are not stored, so resuming initializes fresh
environments and prefills the replay buffer. Historical presets do not reproduce
every interruption or environment-count change of the original runs.

Logs are written to `logs/<run>/train_log.csv` and `stdout.log`. Resource monitoring:

```bash
python tools/monitor_go2.py --run_dir logs/go2_flat --interval 5
```

Training runs on the machine where the command is executed. Use SSH to launch
it on a remote GPU server. `SUPERMIMIC_PYTHON` selects the interpreter, and
`CPUS` optionally sets training CPU affinity through `taskset`.

## Commands and randomization

On flat ground, commands cover forward velocity `[-0.5, 1.0] m/s`, lateral
velocity `[-0.4, 0.4] m/s`, and yaw rate `[-1.0, 1.0] rad/s`. Commands change
every 2-5 seconds, with a 15% probability of a zero command. Non-flat terrain
uses a forward command sampled from `[0.28, 0.50] m/s` once per episode,
with zero lateral velocity and yaw rate. Planar commands use the robot's current
yaw frame. Demonstration evaluations use a fixed forward command of 0.5 m/s.

Training includes robot-side friction parameters in `[0.3, 1.2]`, per-episode
joint calibration offsets within 0.01 rad, initial-state perturbations, and
proprioceptive observation noise. Friction is sampled once per environment;
calibration offsets remain fixed within an episode. Contact inputs use a 5 N
force threshold without simulated sensor errors or latency. Mass, motor
strength, control delays, and external pushes are not randomized.

The baseline presets do not enable the optional action-retention, action-budget,
or real-trajectory value-calibration experiments. These modules remain available
for explicitly configured experiments. No reference-action dataset is included.

## Evaluation and playback

Pretrained checkpoints are not bundled. The filename `checkpoints/go2_51k.pt`
below is a placeholder for a compatible terrain-trained checkpoint.

```bash
# Fixed-command evaluation; command components are vx, vy, and yaw rate.
./run_go2_eval.sh --ckpt checkpoints/go2_51k.pt \
  --terrain_tile stairs_up --command 0.5 0 0 --seconds 20 --visualizer none

# Seven terrain families, levels L1-L5, and eight standard headings.
./run_go2_rough_eval.sh --ckpt checkpoints/go2_51k.pt \
  --output logs/evaluation/rough.json --visualizer none

# View an L5 diagonal stair ascent in real time.
./run_go2_rough_eval.sh --ckpt checkpoints/go2_51k.pt \
  --terrains stairs_up --levels 5 --headings 45 --speed 0.5 \
  --visualizer newton_gl --real_time --output logs/evaluation/stairs_up_l5.json

# Keyboard playback with a 16-step (0.32-second) world-model forecast.
./run_go2_play.sh --ckpt checkpoints/go2_51k.pt \
  --terrain_tile stairs_up --ghost_steps 16

# Compose a continuous route from legacy terrain sections.
./run_go2_course_play.sh --ckpt checkpoints/go2_51k.pt \
  --terrain_sequence flat stairs_up_low stairs_down_low stairs_up stairs_down
```

Keyboard controls: W/S changes forward speed, A/D changes yaw rate, releasing
keys holds the target, Space stops, R resets, F toggles camera following, and
Q/Esc exits. A desktop display and the Newton GL window are required.
Use the rough-evaluation entry point to view individual rough-terrain cases.

Pass `--ckpt` explicitly, or set `SUPERMIMIC_GO2_CKPT`. Playback can also discover
the newest `logs/*/last.pt` in this checkout. Evaluation distinguishes passes,
falls, map exits, and timeouts. The generic rough evaluator starts from the tile
center; specialized historical diagnostics and video-editing scripts are not included.
The fixed-command evaluator supports trajectory export through `--record`.

## Tests

CPU tensor regressions require only Python 3.12, PyTorch, and NumPy:

```bash
python -m unittest tools.test_go2 tools.test_go2_guarded -v
```

Additional checks in the complete pinned runtime:

```bash
python -m unittest tools.test_go2_keyboard tools.test_go2_retention -v
python tools/check_runtime.py
python tools/check_go2.py --visualizer none
```

GitHub Actions runs the CPU suites. These checks verify implementation behavior,
not convergence, general terrain success rates, or real-robot transfer.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE). The original MIT copyright notice
is preserved. Isaac Lab, Newton, MuJoCo, MuJoCo Warp, and robot assets retain
their respective licenses and are not redistributed here.
