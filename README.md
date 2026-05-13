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

**EPO** (Edge-based Pose Optimization) is a featureless method for refining camera poses and depth produced by 3D Foundation Models (3DFMs) such as VGGT. Rather than relying on hand-crafted/learned feature matching, EPO directly optimizes camera poses using dense depth maps, improving reconstruction quality in a lightweight and generalizable manner.



## Changelog

| Version | Description |
|---------|-------------|
| **1.1** | Fixed a bug in the loss function. Outliers are now handled more robustly, resulting in slightly higher scores. |
| **1.0** | Initial release. |



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

pip install h5py \
            kornia \
            matplotlib \
            numpy \
            opencv-python \
            pandas \
            pycolmap==3.11 \
            scikit-learn \
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

pip install h5py \
            kornia \
            matplotlib \
            numpy \
            opencv-python \
            pandas \
            pycolmap==3.11 \
            scikit-learn \
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

Organize your images using the following structure. Images can be grouped by camera if multiple camera rigs are used:

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
    └── depth_maps/
        ├── 1/               # Depth maps for Camera Group 1
        │   ├── _DSC8679.h5
        │   ├── _DSC8680.h5
        │   └── ...
        └── 2/               # Depth maps for Camera Group 2 (if more)
            ├── _DSC9001.h5
            └── ...
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