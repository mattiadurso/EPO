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

# gsplat 1.5.3 forces C++17, but torch >= 2.14's headers need C++20.
if ! "$PY" -c "import sys, torch; sys.exit(tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 14))"; then
  echo "ReSplat's gsplat 1.5.3 cannot be built against torch $("$PY" -c 'import torch; print(torch.__version__)')." >&2
  echo "Use torch < 2.14 (environment.yml pins 2.11), e.g. pip install 'torch<2.14' 'torchvision<0.29'." >&2
  exit 1
fi

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
#    ninja parallelizes the compile (pip's copy if there is none on PATH).
command -v ninja >/dev/null || "$PY" -m pip install ninja
"$PY" -m pip install --no-build-isolation --no-deps --no-binary gsplat gsplat==1.5.3
cp -r "$RESPLAT/src/model/encoder/pointops" "$BUILD/pointops"
rm -rf "$BUILD/pointops/build" "$BUILD"/pointops/*.egg-info
"$PY" -m pip install --no-build-isolation --no-deps "$BUILD/pointops"

# 4. Pure-Python deps imported by ReSplat's config / model modules: the
#    missing ones of pyproject.toml's [resplat] extra.
"$PY" -c "import sys; sys.path.insert(0, '$ROOT'); \
from wrapper.deps import ensure_extra; ensure_extra('resplat')"

"$PY" -c "import gsplat, pointops, e3nn; print('ReSplat ready: demo_epo.py --3dgsfy')"
