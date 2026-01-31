import os
import time
import json
import argparse
from adjuster import Adjuster

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process dataset scenes")
parser.add_argument("--dataset", type=str, default="mipnerf360", help="Dataset name")
parser.add_argument("--edges", type=str, default="canny", help="Edge type")
parser.add_argument("--note", type=str, default="", help="Run note")
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
    scenes = sorted(os.listdir(dataset_cfg["base_path"]), reverse=False)

    # remove = ["stump", "treehill"]  # unwanted folders/files
    # scenes = [s for s in scenes if s not in remove]

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

        opt = f"benchmarks/vggt_edge_{args.edges}{args.note}/{dataset}/{scene}/sparse"

        os.makedirs(opt, exist_ok=True)

        # ==============================================================================
        #                     Adjuster
        # ==============================================================================

        adjuster = Adjuster(
            reconstruction_path=reconstruction_path,
            images_path=images_path,
            depths_path=depths_path,
            grad_q=True,
            grad_t=True,
            grad_t_offset=False,
            grad_k=True,
            grad_z=True,
            use_mlp_pose_refinement=True,
            q_lr=1e-4,
            t_lr=1e-3,
            k_lr=1e-3,
            z_lr=3e-3,
            mlp_pose_lr=3e-3,
            max_edges_points=12_288,
            max_viewgraph_pairs=4_096,
            single_camera_per_folder=True,
            max_num_iterations=100,
            viz=True,
            verbose=False,
        )

        # run the optimization
        adjuster(
            convergence_tol_pose=0.5,  # degrees
            convergence_tol_depth=0.1,  # relative change %
            convergence_tol_loss=5e-5,  # relative change %
            window_loss=50,
            # gt_path=gt_path,
            early_stop="none",  # to stop after second pose convergence
        )

        adjuster.to_colmap(
            opt,
            verbose=False,
            max_points_per_image=100_000 // len(adjuster.images),
            save_points=True,
            final_dbscan_filtering=False,
        )


print(f"Total time: {time.time() - s_time:.2f} seconds")

# python test.py mipnerf360 --edges canny && python test.py terrasky3D --edges canny && python test.py scannetpp --edges canny
# python test.py mipnerf360 --edges teed && python test.py terrasky3D --edges teed && python test.py scannetpp --edges teed
