import os
import time
import json
import argparse
from adjuster import Adjuster

parser = argparse.ArgumentParser(description="Process dataset scenes")
parser.add_argument("dataset", type=str, default="mipnerf360", help="Dataset name")
args = parser.parse_args()


# Load dataset paths and parameters from JSON
with open("benchmark/paths.json") as f:
    paths_cfg = json.load(f)

dataset = args.dataset  # Change this to switch dataset
dataset_cfg = paths_cfg[dataset]

# Get scenes list
if "scenes" in dataset_cfg:
    scenes = dataset_cfg["scenes"]
else:
    # If scenes not listed, use folders in base_path
    scenes = sorted(os.listdir(dataset_cfg["base_path"]))

s_time = time.time()
for scene in scenes:
    # Compose paths using template
    images_path = os.path.join(
        dataset_cfg["images_path"], scene, dataset_cfg["images_folder"]
    )
    reconstruction_path = os.path.join(
        dataset_cfg["base_path"], scene, dataset_cfg["reconstruction_folder"]
    )
    depths_path = os.path.join(
        dataset_cfg.get("base_path", dataset_cfg.get("images_path", "")),
        scene,
        dataset_cfg.get("depths_folder", dataset_cfg.get("depth_folder", "")),
    )
    gt_path = os.path.join(dataset_cfg["gt_path"], scene, dataset_cfg["gt_folder"])

    # ==============================================================================
    # Adjuster
    # ==============================================================================
    adjuster = Adjuster(
        # paths
        reconstruction_path=reconstruction_path,
        images_path=images_path,
        depths_path=depths_path,
        unreliable_area_masks_path=images_path.replace(
            dataset_cfg["images_folder"], "depth_masks_mask2former"
        ),
        # intrinsics
        single_camera_per_folder=True,
        # detect and match
        detector="teed",
        matcher_type="frustum",  # or "exhaustive", "sequential"
        scheduler_params={"factor": 0.75, "patience": 3, "min_lr": 1e-5},
        # optimization
        lr=5e-4,
        grad_z=True,
        grad_q=True,
        grad_k=True,
        grad_t=True,
    )

    adjuster(batch_size=256, max_steps=-1, residuals_chunk_size=2048, debug=False)

    opt = f"benchmark/vggt_edge/{dataset}/{scene}/sparse"
    os.makedirs(opt, exist_ok=True)

    adjuster.to_colmap(
        opt,
        verbose=False,
        save_points=False,
    )


print(f"Total time: {time.time() - s_time:.2f} seconds")
