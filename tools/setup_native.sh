#!/bin/bash
# Install the pinned kit-less runtime into a dedicated Conda environment.
# Optional: SUPERMIMIC_ENV, SUPERMIMIC_ISAACLAB_DIR, SUPERMIMIC_CONDA_CHANNEL.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
ENV_NAME=${SUPERMIMIC_ENV:-supermimic-go2}
LAB=${SUPERMIMIC_ISAACLAB_DIR:-$REPO/IsaacLab}
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
CONDA_BIN=${CONDA_EXE:-$(command -v conda || true)}
if [ -z "$CONDA_BIN" ]; then
  echo "Install Conda and make its conda executable available on PATH first." >&2
  exit 1
fi
for tool in git gcc g++; do
  command -v "$tool" >/dev/null || { echo "Missing native build tool: $tool" >&2; exit 1; }
done

if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys' >/dev/null 2>&1; then
  CHANNEL=()
  if [ -n "${SUPERMIMIC_CONDA_CHANNEL:-}" ]; then
    CHANNEL=(--override-channels -c "$SUPERMIMIC_CONDA_CHANNEL")
  fi
  "$CONDA_BIN" create -n "$ENV_NAME" python=3.12 pip -y "${CHANNEL[@]}"
fi
ENV_PYTHON=$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
"$ENV_PYTHON" -m pip install 'uv==0.12.23'
UV="$(dirname "$ENV_PYTHON")/uv"

if git -C "$LAB" rev-parse --verify HEAD >/dev/null 2>&1; then
  if [ "$(git -C "$LAB" rev-parse HEAD)" != "$ISAACLAB_COMMIT" ]; then
    echo "$LAB is not at $ISAACLAB_COMMIT; use a separate SUPERMIMIC_ISAACLAB_DIR." >&2
    exit 1
  fi
else
  mkdir -p "$LAB"
  git -C "$LAB" init
  if ! git -C "$LAB" remote get-url origin >/dev/null 2>&1; then
    git -C "$LAB" remote add origin https://github.com/isaac-sim/IsaacLab.git
  fi
  git -C "$LAB" fetch --depth=1 origin "$ISAACLAB_COMMIT"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$LAB" checkout --detach FETCH_HEAD
fi

mkdir -p "$REPO/.cache"
cd "$LAB"
# Export the upstream lock rather than re-resolving its dependencies. Install
# into Conda with uv pip; uv sync would manage/replace its own virtualenv.
"$UV" export --frozen --extra importers --extra test --no-dev --no-hashes \
  --format requirements-txt --output-file "$REPO/.cache/native-requirements.txt" >/dev/null
# Every dependency is already pinned by the lock. --no-deps also preserves the
# upstream overrides required by the standalone USD importers and Newton.
CMAKE_POLICY_VERSION_MINIMUM=3.5 "$UV" pip install --python "$ENV_PYTHON" --no-deps \
  --index-url https://pypi.org/simple --index https://download.pytorch.org/whl/cu130 \
  --index https://pypi.nvidia.com --index-strategy unsafe-best-match \
  -r "$REPO/.cache/native-requirements.txt"
cd "$REPO"
"$UV" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation -e "$REPO"
"$ENV_PYTHON" tools/check_runtime.py
echo "Ready: conda activate $ENV_NAME"
