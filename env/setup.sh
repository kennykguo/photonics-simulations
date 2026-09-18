#!/usr/bin/env bash
# Reproduces the environment used by every experiment (macOS, Apple silicon).
# 1. Homebrew tools: ffmpeg (video encoding), ngspice (circuit simulator), micromamba (conda-forge installer for Meep)
# 2. Python 3.12 venv via uv with the scientific + photonics stack
# 3. A separate conda-forge environment for Meep (pymeep has no pip wheel)
set -euo pipefail
cd "$(dirname "$0")/.."
brew install ffmpeg ngspice micromamba uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  numpy scipy matplotlib jupyterlab ipywidgets ipympl sympy scikit-rf control plotly pandas pillow imageio imageio-ffmpeg \
  manim femwell tidy3d sax gdsfactory fdtd jax jaxlib meshio gmsh shapely scikit-fem h5py tqdm nbformat nbclient
export MAMBA_ROOT_PREFIX="$PWD/.mamba"
micromamba create -y -p "$PWD/.meep" -c conda-forge python=3.12 pymeep numpy scipy matplotlib h5py imageio imageio-ffmpeg
echo "done. Python: .venv/bin/python   Meep: .meep/bin/python   ngspice: $(which ngspice)"
