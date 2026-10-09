#!/usr/bin/env bash
# Reproduces this folder's environment from scratch on a fresh Mac.
#
#   bash setup.sh
#
# Minimal environment for visuals/ only (env/setup.sh at the repo root installs
# the full lab). Installs into ../.venv only. Nothing is installed globally except uv itself,
# and only if it is missing. No Homebrew packages are needed: the video encoder
# comes from the imageio-ffmpeg wheel.
set -euo pipefail
cd "$(dirname "$0")/.."   # the venv lives at the repo root, shared with experiments/

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, installing it"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "creating .venv on Python 3.12"
uv venv --python 3.12 .venv

echo "installing packages"
uv pip install --python .venv/bin/python \
  numpy scipy matplotlib imageio-ffmpeg femwell pyvista plotly

echo
echo "checking the install"
.venv/bin/python - <<'PY'
import numpy, scipy, matplotlib, imageio_ffmpeg, femwell, skfem, gmsh, shapely
print("  numpy      ", numpy.__version__)
print("  scipy      ", scipy.__version__)
print("  matplotlib ", matplotlib.__version__)
print("  femwell    ", "installed (FEM mode solver)")
print("  scikit-fem ", skfem.__version__)
print("  gmsh       ", gmsh.__version__ if isinstance(gmsh.__version__, str) else "installed")
print("  ffmpeg     ", imageio_ffmpeg.get_ffmpeg_exe())
PY

echo
echo "done. now see README.md for the run command of each visual"
