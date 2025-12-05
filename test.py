import os
import time
import json
import argparse
from adjuster import Adjuster

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process dataset scenes")
parser.add_argument("dataset", type=str, default="mipnerf360", help="Dataset name")
args = parser.parse_args()


# Load dataset paths and parameters from JSON
with open("benchmarks/paths.json") as f:
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
    #                     Adjuster
    # ==============================================================================

    adjuster = Adjuster(
        reconstruction_path=reconstruction_path,
        images_path=images_path,
        depths_path=depths_path,
        q_lr=1e-4,
        grad_q=False,
        t_lr=1e-3,
        grad_t=False,
        k_lr=1e-3,
        grad_k=False,
        z_lr=1e-3,
        grad_z=False,
        mlp_pose_lr=1e-3,
        use_mlp_pose_refinement=True,
        detector="canny",  # or "canny", "bdcn", "sam2"
        detector_params={
            "low_threshold": 0.20,
            "high_threshold": 0.25,
            "kernel_size": 7,
            "sigma": 2,
        },
        matcher_type="exhaustive",  # or "exhaustive", "sequential", "frustums"
        viz=True,
        use_amp=False,
    )

    adjuster(batch_size=512, max_steps=1_000, residuals_chunk_size=2048, debug=False)

    opt = f"benchmarks/vggt_edge/{dataset}/{scene}/sparse"
    os.makedirs(opt, exist_ok=True)

    adjuster.to_colmap(
        opt,
        verbose=False,
        save_points=False,
    )


print(f"Total time: {time.time() - s_time:.2f} seconds")

# python test.py mipnerf360 && python test.py terrasky3D && python test.py tt
