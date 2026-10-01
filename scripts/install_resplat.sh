#!/usr/bin/env bash
# One-time setup of ReSplat (feed-forward 3DGS) for `demo_epo.py --3dgsfy`,
# inside the EPO env:
#
#   conda activate epo && bash scripts/install_resplat.sh
#
# Fetches third_party/resplat, applies EPO's many-views patch, installs
# NVIDIA's pip nvcc matching torch's CUDA (no system toolkit needed), builds
# gsplat + pointops against the env's torch and adds ReSplat's small
# pure-Python deps (none of them changes an existing EPO package). The
# weights download from the Hugging Face Hub on first use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESPLAT="$ROOT/third_party/resplat"
PATCH="$ROOT/third_party/patches/resplat_many_views.patch"
PY="${PYTHON:-python}"

# 1. Source + patch: chunks the cost volume, the point transformer and the
#    renderer over views/points so ~150 views fit in 24 GB (same maths).
git -C "$ROOT" submodule update --init third_party/resplat
if git -C "$RESPLAT" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "ReSplat patch already applied"
else
  git -C "$RESPLAT" apply "$PATCH"
fi

# 2. nvcc from the same CUDA release as torch's pip wheels.
TOOLKIT=$("$PY" -c "import importlib.metadata as m; print(m.version('cuda-toolkit'))")
"$PY" -m pip install "cuda-toolkit[nvcc,cccl,crt,nvvm]==$TOOLKIT"
MAJOR=$("$PY" -c "import torch; print(torch.version.cuda.split('.')[0])")
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export CUDA_HOME="$SITE/nvidia/cu$MAJOR"
export PATH="$CUDA_HOME/bin:$PATH"
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
# The linker wants the unversioned name, which the wheels do not ship.
ln -s "$CUDA_HOME/lib/libcudart.so.$MAJOR" "$BUILD/libcudart.so"
export LIBRARY_PATH="$BUILD${LIBRARY_PATH:+:$LIBRARY_PATH}"
ARCH=$("$PY" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}"

# 3. CUDA extensions. gsplat 1.5.3 is the version the EPO numbers used;
#    pointops is built from a clean copy so no stale objects get reused.
"$PY" -m pip install --no-build-isolation --no-deps --no-binary gsplat gsplat==1.5.3
cp -r "$RESPLAT/src/model/encoder/pointops" "$BUILD/pointops"
rm -rf "$BUILD/pointops/build" "$BUILD"/pointops/*.egg-info
"$PY" -m pip install --no-build-isolation --no-deps "$BUILD/pointops"

# 4. Pure-Python deps imported by ReSplat's config / model modules.
"$PY" -m pip install dacite==1.8.1 e3nn==0.5.1 sk-video==1.1.10 lpips==0.1.4 \
  colorspacious==1.1.2

"$PY" -c "import gsplat, pointops, e3nn; print('ReSplat ready: demo_epo.py --3dgsfy')"
