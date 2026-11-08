#!/bin/bash
# Native Go2 velocity training.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-go2_velocity}
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid run name: use letters, numbers, '.', '_' or '-'." >&2
  exit 2
fi
PYTHON=${SUPERMIMIC_PYTHON:-python}
if [ -z "${ISAACSIM_ASSET_ROOT:-}" ] && [ -d "$PWD/data/go2_isaac_assets" ]; then
  export ISAACSIM_ASSET_ROOT="$PWD/data/go2_isaac_assets"
fi
PIN=()
if [ -n "${CPUS:-}" ]; then PIN=(taskset -c "$CPUS"); fi
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$PWD/.cache/warp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PWD/.cache/inductor}
export PYTHONUNBUFFERED=1
mkdir -p "logs/$RUN"
"${PIN[@]}" "$PYTHON" -m supermimic.go2.train --log_dir "logs/$RUN" "$@" 2>&1 | tee -a "logs/$RUN/stdout.log"
