<div align="center">

# Boosting 3D Foundation Models with Featureless Pose Optimization

<p>
  <a href="https://scholar.google.com/citations?user=9FjTo3YAAAAJ&hl=en">Mattia D'Urso</a><sup>1</sup>&ensp;·&ensp;
  <a href="https://scholar.google.com/citations?user=6uZVF04AAAAJ&hl=en">Christian Sormann</a><sup>2</sup>&ensp;·&ensp;
  <a href="https://scholar.google.com/citations?user=DA3nSvgAAAAJ&hl=en">Mattia Rossi</a><sup>2</sup>&ensp;·&ensp;
  <a href="https://scholar.google.com/citations?user=M0boL5kAAAAJ&hl=en">Friedrich Fraundorfer</a><sup>1</sup>
</p>

<p>
  <sup>1</sup>Graz University of Technology&emsp;·&emsp;<sup>2</sup>Sony Europe
</p>

<p>
  👀
</p>

<p>
  <a href=""><img src="https://img.shields.io/badge/Paper-2026-blue?style=flat-square" alt="Paper"></a>
  &nbsp;
  <a href=""><img src="https://img.shields.io/badge/Supplementary-Material-green?style=flat-square" alt="Supplementary Material"></a>
  &nbsp;
  <a href="https://mattiadurso.github.io/epo/"><img src="https://img.shields.io/badge/Project-Page-orange?style=flat-square" alt="Project Page"></a>
</p>



<img src="assets/townhall.gif" alt="EPO Pose Optimization" width="480px">

<p><em>
  Visualization of three stages of EPO applied to the Graz Town Hall scene (TerraSky3D). Starting from the initial state <strong>(a)</strong> provided by VGGT output, we show an intermediate step <strong>(b)</strong> and the final refined poses <strong>(c)</strong>. Ground truth poses are shown in <span style="color:green">green</span>; optimized poses in <span style="color:red">red</span>.
</em></p>

</div>


## Overview

**EPO** (Edge-based Pose Optimization) is a trackless method for refining camera poses and depth produced by 3D Foundation Models (3DFMs) such as VGGT. Rather than relying on hand-crafted/learned feature matching, EPO directly optimizes camera poses and depth maps using edges reprojection, improving reconstruction quality in a lightweight and generalizable manner.



## Changelog

| Version | Description |
|---------|-------------|
| **1.0.3** | Fused the Huber loss epilogue into the Triton kernel (bit-identical, faster); optional `fuse_reduction=True` fuses the loss reduction too (faster, fp-reordering only).<br>Per-step logging now syncs GPU→CPU once per iteration; `log_granular_time` now defaults to `False`.<br>Torch backend now masks behind-camera/non-finite points identically to Triton. |
| **1.0.2** | Added `EPO.from_ff(...)` — initialize directly from a 3DFM's in-memory output (no COLMAP/`.h5` round-trip). |
| **1.0.1** | Fixed a bug in the loss function — outliers are now handled more robustly, resulting in slightly higher scores.<br>Added Triton kernel for point reprojection (~1.5× faster); enable with `backend="triton"`.<br>Added mixed-precision (BF16) support for the pose-refinement MLP via `use_amp=True`.<br>Improved per-stage time logging; can be disabled with `log_granular_time=False`. |
| **1.0.0** | Initial release. |



## Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU
- One of: [Conda](https://docs.conda.io/en/latest/) **or** `python3-venv` (any pip-only flow)

### Option A — Conda Environment File

```bash
git clone https://github.com/mattiadurso/epo.git
cd epo

conda env create -f environment.yml
conda activate epo
```

### Option B — Conda + Manual Pip Install

```bash
conda create -n epo python=3.10 -y
conda activate epo

pip install kornia \
            matplotlib \
            numpy \
            opencv-python \
            pandas \
            pycolmap \
            torch \
            torchvision \
            triton \
            tqdm \
            git+https://github.com/mattiadurso/mylib.git
```

### Option C — Pip + venv (no Conda required)

```bash
git clone https://github.com/mattiadurso/epo.git
cd epo

python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install kornia \
            matplotlib \
            numpy \
            opencv-python \
            pandas \
            pycolmap==3.11 \
            torch \
            torchvision \
            triton \
            tqdm \
            git+https://github.com/mattiadurso/mylib.git
```

> ℹ️ `triton` is Linux-only and requires a CUDA build of `torch`. On systems without CUDA, install `torch` from the [official selector](https://pytorch.org/get-started/locally/) first, then run the rest of the `pip install` line without `triton` — EPO will fall back to the PyTorch reference path (`backend="torch"`).

## Usage

### Step 1 — Prepare Input Images

> 📦 A few demo scenes can be downloaded [here](https://cloud.tugraz.at/index.php/s/dNfD96WNJ64kZCS).

Organize your images using the following structure. Images can be grouped by camera if multiple cameras are used:

```
bicycle/
└── images/
    ├── 1/                   # Images for Camera Group 1
    │   ├── _DSC8679.jpg
    │   ├── _DSC8680.jpg
    │   └── ...
    └── 2/                   # Images for Camera Group 2 (if more)
        ├── _DSC9001.jpg
        └── ...
```

### Step 2 — Run a 3D Foundation Model

Run a 3DFM (e.g., [VGGT](https://github.com/mattiadurso/vggt)) on your images and export the reconstruction in **COLMAP format** and the dense depth maps with the following layout:

```
bicycle/
└── sparse/
    ├── cameras.bin          # Camera intrinsic parameters
    ├── images.bin           # Camera extrinsics and image registration
    ├── points3D.bin         # Sparse 3D point cloud
    └── depths.pth           # Dense depth maps: torch.save'd dict
                             # {image_stem: {"depth": (H, W), "confidence": (H, W) (optional)}}
```



### Step 3 — Run EPO

```bash
python epo.py \
    --images_path <path_to_images> \
    --rec_path    ./sparse \
    --depth_path  ./depths \
    --gt_path     <path_to_ground_truth>   # Optional — enables quantitative evaluation
```

### Notes

- Image and depth map directories must mirror the same folder structure.
- Ground truth data must be provided in COLMAP format, with matching image names (e.g., `cam/images.jpg`).

### Programmatic use — feed-forward output 

If you already have a 3DFM's output as tensors in memory, you can skip the COLMAP / `depths.pth` export and feed EPO directly:

```python
from epo import EPO

# ff_data: dict keyed by "cam_id/image_name"
# Each value is a dict with the per-image tensors (already at images_size).
# Pose is world-to-camera (T_cw), matching COLMAP / PoseModule.
ff_data = {
    "cam0/img_000.jpg": {
        "image":     image_tensor,      # (3, H, W) float in [0, 1]
        "depth":     depth_tensor,      # (H, W)
        "pose":      pose_tensor,       # (3, 4) or (4, 4), world-to-camera
        "intrinsic": intrinsic_tensor,  # (3, 3) pinhole
        # "confidence": conf_tensor,    # (H, W), optional
    },
    ...
}

epo = EPO.from_ff(
    ff_data,
    backend="triton",
)
epo()
epo.to_colmap("out/sparse", save_points=True)
```

All other `EPO(...)` kwargs are forwarded. Images and depths must already be at `images_size` (no internal resize). One camera per image.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{durso2026epo,
  title     = {Boosting 3D Foundation Models with Featureless Pose Optimization},
  author    = {Mattia D'Urso and Christian Sormann and Mattia Rossi and Friedrich Fraundorfer},
  booktitle = {👀},
  year      = {2026},
}
```

---

<div align="center">
  <sub>Graz University of Technology · Sony Europe · 2026</sub>
</div>