"""Batch runner for EPO across one or more benchmark datasets.

Reads dataset roots from ``benchmarks/paths.json`` and, for every scene
discovered under each dataset, runs the EPO optimization and exports the
refined reconstruction in COLMAP format under ``benchmarks/<run_name>/``.

Examples:
    python test.py --dataset all       --edges canny --early_stop loss
    python test.py --dataset terrasky3D --edges canny --max_iterations 1000
"""

import os
import time
import json
import argparse
from epo import EPO

parser = argparse.ArgumentParser(description="Run EPO across benchmark scenes")
parser.add_argument("--dataset", type=str, default="all", help="Dataset name")
parser.add_argument("--edges", type=str, default="canny", help="Edge type")
parser.add_argument("--note", type=str, default="", help="Run note")
parser.add_argument(
    "--max_iterations", type=int, default=2000, help="Maximum number of iterations"
)
parser.add_argument(
    "--early_stop", type=str, default="pose", help="Early stopping criterion"
)
parser.add_argument("--model", type=str, default="vggt", help="Model name")
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

    # Each subdirectory of the dataset root is treated as one scene.
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

        note = "_" + args.note if args.note != "" else ""
        opt = (
            f"benchmarks/{args.model}_edge_{args.edges}{note}/{dataset}/{scene}/sparse"
        )

        if not os.path.isdir(reconstruction_path):
            print(f"Skipping {reconstruction_path}")
            continue

        os.makedirs(opt, exist_ok=True)

        if args.model != "vggt":
            reconstruction_path = reconstruction_path.replace("vggt", args.model)
            depths_path = depths_path.replace("vggt", args.model)

        # --- Build and run EPO for this scene ---
        epo = EPO(
            reconstruction_path=reconstruction_path,
            images_path=images_path,
            depths_path=depths_path,
            grad_t_offset=True,
            grad_k=True,
            grad_z=True,
            use_mlp_pose_refinement=True,
            detector=args.edges,
            detector_params={
                "low_threshold": 0.15,
                "high_threshold": 0.20,
                "kernel_size": 9,
                "sigma": 2,
            },
            k_lr=3e-3,
            z_lr=3e-3,
            mlp_pose_lr=3e-3,
            max_edges_points=12_288,
            max_viewgraph_pairs=4_096,
            single_camera_per_folder=True,
            max_num_iterations=args.max_iterations,
            verbose=True,
            # Viewgraph parameters
            min_points=750,
            sampling_factor=5,
            reprojection_error=3,
            auc_saving_freq=100,
            use_amp=True,  # Whether to run the pose-refinement MLP's linear layers in BF16 via torch.autocast. Gram-Schmidt stays in FP32 (precision-sensitive). No GradScaler needed for BF16. Default False (FP32 throughout — historical behaviour).
            backend="triton",  # fused Triton kernels for both project+sample and unproject (fwd + analytical bwd). Much faster than the PyTorch path on large viewgraphs / many edges.
        )

        epo(
            window_pose=25,
            window_depth=50,
            convergence_tol_pose=0.5,  # degrees
            convergence_tol_depth=0.1,  # relative change (%)
            early_stop=args.early_stop,
        )

        epo.to_colmap(
            opt,
            verbose=False,
            max_points_per_image=100_000 // len(epo.images),
            save_points=True,
            final_dbscan_filtering=False,
        )


print(f"Total time: {time.time() - s_time:.2f} seconds")
