#!/bin/bash
# Start keyboard-controlled Go2 playback in the native Conda environment.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${SUPERMIMIC_PYTHON:-python}
if [ -z "${ISAACSIM_ASSET_ROOT:-}" ] && [ -d "$PWD/data/go2_isaac_assets" ]; then
  export ISAACSIM_ASSET_ROOT="$PWD/data/go2_isaac_assets"
fi
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$PWD/.cache/warp}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
exec "$PYTHON" -m supermimic.go2.play "$@"
