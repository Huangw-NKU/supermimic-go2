#!/bin/bash
# Play a terrain checkpoint on the combined course; pass --ckpt explicitly.
set -euo pipefail
cd "$(dirname "$0")"

exec ./run_go2_play.sh \
  --terrain_tile mixed --ghost_steps 16 "$@"
