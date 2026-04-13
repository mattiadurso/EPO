<p align="center">
  <h1 align="center">EPO: Boosting 3D Foundation Models with Featureless Pose Optimization</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=9FjTo3YAAAAJ&hl=en">Mattia D'Urso</a><sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?user=6uZVF04AAAAJ&hl=en">Christian Sormann</a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=DA3nSvgAAAAJ&hl=en">Mattia Rossi</a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=M0boL5kAAAAJ&hl=en">Friedrich Fraundorfer</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Graz University of Technology · <sup>2</sup> Sony Europe
  </p>
  <h2 align="center">
    <p>👀 2026</p>
    <a href="" align="center">Paper</a> | 
    <a href="" align="center">Supplementary Material</a> | 
    <a href="https://mattiadurso.github.io/epo/" align="center"> Project Page</a>
  </h2>
</p>

<p align="center">
  <img src="assets/townhall.gif" alt="ALIKED" width="400px">
  <br>
  <em>Visualization of three stages of our pose optimization method and the ground truth sparse point cloud for the Graz Town Hall scene (TerraSky3D). Starting from the initial optimization stage (a), provided by the VGGT output, we illustrate an intermediate state (b) and the final refined poses (c). The ground truth poses are shown in green and the optimized ones in red.</em>
</p>

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mattiadurso/epo.git
cd epo
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml     # not there yet, but dependencies are minimal
conda activate epo
```


## How to run

### Quick Start

Run a 3DFM (e.g., VGGT) on an unconstrained set of RGB images 
```
bicycle/
└── images/                     # Images folder
    ├── 1/                      # Images for Camera Group 1
    │   ├── _DSC8679.jpg         
    │   ├── _DSC8680.jpg
    │   └── ...
    └── 2/                      # Images for Camera Group 2 (if applicable)
        ├── _DSC9001.jpg
        └── ...
```

and save the input in COLMAP format with the following format. A demo scene can be downloaded <a href="https://cloud.tugraz.at/index.php/s/dNfD96WNJ64kZCS">here</a>.

```
bicycle/
└── sparse/                         # VGGT reconstruction (COLMAP format)
    ├── cameras.bin                 # Camera intrinsics parameters
    ├── images.bin                  # Camera poses and registration
    ├── points3D.bin                # Sparse 3D point cloud data
    └── depth_maps/                 # VGGT dense depth maps
        ├── 1/                      # Depth maps for Camera Group 1
        │   ├── _DSC8679.h5         # Per-pixel depth data (HDF5)
        │   ├── _DSC8680.h5
        │   └── ...
        └── 2/                      # Depth maps for Camera Group 2 (if applicable)
            ├── _DSC9001.h5
            └── ...
```

Then run with
```bash
python epo.py \
    --images_path <path> \    # Path to original source images
    --rec_path ./sparse \     # COLMAP reconstruction folder (.bin files)
    --depth_path ./depths \   # Folder containing the .h5 depth maps
    --gt_path <path>          # (Optional) Ground truth data for evaluation
```
Notice: 
- Images and depth maps must mirror the same file system structure.
- Ground truth must be in the same COLMAP format, including images names (e.g. cam/images.jpg)

## Citation

```bibtex
@inproceedings{durso2026epo,
  title={Boosting 3D Foundation Models with Featureless Pose Optimization},
  author={Mattia D'Urso and Christian Sormann and Mattia Rossi and Friedrich Fraundorfer},
  booktitle={},
  year={2026},
  url={}
  }
```
