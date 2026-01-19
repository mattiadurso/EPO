import os
import time
import json
import argparse
from adjuster import Adjuster

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process dataset scenes")
parser.add_argument("dataset", type=str, default="terrasky3D", help="Dataset name")
parser.add_argument("--edges", type=str, default="canny", help="Edge type")
args = parser.parse_args()


# Load dataset paths and parameters from JSON
with open("benchmarks/paths.json") as f:
    paths_cfg = json.load(f)

if args.dataset == "all":
    datasets = ["mipnerf360", "terrasky3D", "scannetpp"]
else:
    datasets = [args.dataset]

s_time = time.time()
for dataset in datasets:
    dataset_cfg = paths_cfg[dataset]

    ## Get scenes list
    scenes = sorted(os.listdir(dataset_cfg["base_path"]))

    print(
        f"Processing dataset '{dataset}' with {len(scenes)} scenes: {scenes}",
        end="\n\n",
    )

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
            # q_lr=1e-4,
            # grad_q=True,
            # t_lr=1e-3,
            # grad_t=True,
            k_lr=1e-3,
            grad_k=True,
            z_lr=3e-3,
            grad_z=True,
            mlp_pose_lr=3e-3,
            use_mlp_pose_refinement=True,
            detector=args.edges,  # or "canny", "bdcn", "sam2"
            detector_params={
                "low_threshold": 0.20,
                "high_threshold": 0.25,
                "kernel_size": 7,
                "sigma": 2,
            },
            matcher_type="exhaustive",  # or "exhaustive", "sequential", "frustums"
            viz=True,
            use_amp=False,
            max_edges_points=12_288,
            max_viewgraph_pairs=4_096,
            single_camera_per_folder=True,
            max_num_iterations=2_000,
        )

        # run the optimization
        adjuster(
            batch_size=256,
            residuals_chunk_size=2048,
            debug=False,
            window_pose=25,
            window_depth=50,
        )

        opt = f"benchmarks/vggt_edge_{args.edges}/{dataset}/{scene}/sparse"
        os.makedirs(opt, exist_ok=True)

        adjuster.to_colmap(
            opt,
            verbose=False,
            save_points=False,
        )


print(f"Total time: {time.time() - s_time:.2f} seconds")

# python test.py mipnerf360 --edges canny && python test.py terrasky3D --edges canny && python test.py scannetpp --edges canny
# python test.py mipnerf360 --edges teed && python test.py terrasky3D --edges teed && python test.py scannetpp --edges teed
