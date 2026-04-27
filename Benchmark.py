import os
import time
import random
import argparse
import numpy as np

from adjuster import Adjuster
from tqdm import tqdm
from evo.tools import file_interface
from evo.core import sync, metrics


def image_to_ts(ts_path):
    with open(ts_path, "r") as f:
        data = f.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")

    name_to_ts = {}
    for l in lines:
        comp = l.split(" ")
        name_to_ts[f"1/{comp[-1]}"] = comp[0]

    return name_to_ts


def eval_tum(ts_est_path, ts_gt_path):
    traj_gt  = file_interface.read_tum_trajectory_file(ts_gt_path)
    traj_est = file_interface.read_tum_trajectory_file(ts_est_path)
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    traj_est.align(traj_gt, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_gt, traj_est))
    return ape.get_all_statistics()


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default=None, help="Dataset to run ['TUM_RGBD', 'Scannet', 'Replica']")
    parser.add_argument("-s", "--scene",   default=None, help="Scene to run (see benchmarks/paths.json)")
    parser.add_argument("-l", "--log",     default="./benchmark_results.json", help="Path to save benchmark log")
    parser.add_argument("-w", "--window",  default=100,   type=int, help="Number of frames for the mapping init phase")
    parser.add_argument("--opt-interval",  default=350,  type=int, help="Run global refinement every N tracking frames")
    args = parser.parse_args()

    print(f"Mapping window: {args.window} frames  |  Refinement interval: {args.opt_interval} frames")

    with open("benchmarks/paths.json") as f:
        paths_cfg = json.load(f)

    datasets = ["TUM_RGBD", "Scannet", "Replica"] if args.dataset is None else [args.dataset]

    ape_per_dataset  = []
    time_per_dataset = []
    scenes_data = {}

    for dataset in datasets:

        dataset_cfg = paths_cfg[dataset]
        base_path   = dataset_cfg["base_path"]
        scenes      = sorted(os.listdir(base_path))

        if args.scene is not None:
            if args.scene not in scenes:
                raise ValueError(f"Scene '{args.scene}' not found in {dataset}.")
            scenes = [args.scene]

        scene_rmse = []
        scene_time = []
        scenes_data[dataset] = {}

        for scene in scenes:

            if not os.path.isdir(os.path.join(base_path, scene)):
                continue

            images_path         = os.path.join(base_path, scene, dataset_cfg["images_folder"])
            reconstruction_path = os.path.join(base_path, scene, dataset_cfg["reconstruction_folder"])
            depths_path         = os.path.join(base_path, scene, dataset_cfg.get("depths_folder", dataset_cfg.get("depth_folder", "")))
            gt_folder           = dataset_cfg.get("gt_folder")
            gt_path             = os.path.join(base_path, scene, gt_folder) if gt_folder else None

            # ----------------------------------------------------------------
            # Mapping phase
            # ----------------------------------------------------------------
            adjuster = Adjuster(
                reconstruction_path=reconstruction_path,
                images_path=images_path,
                depths_path=depths_path,

                k_lr=3e-3,
                z_lr=3e-3,
                grad_k=True,
                grad_z=True,
                grad_t_offset=False,
                use_mlp_pose_refinement=True,
                mlp_pose_lr=3e-3,
                detector="canny",
                detector_params={"low_threshold": 0.20, "high_threshold": 0.25, "kernel_size": 7, "sigma": 2},
                matcher_type="exhaustive",
                viz=True,
                verbose=False,
                max_edges_points=1024 * 12,
                max_viewgraph_pairs=1024 * 6,
                single_camera_per_folder=True,
                auc_saving_freq=50,
                max_num_iterations=2000,
                scale = False,
                n_mapping_frames=args.window,
            )

            t0 = time.time()
            adjuster(
                window_pose=25,
                window_depth=50,
                window_loss=100,
                convergence_tol_pose=0.5,
                convergence_tol_depth=0.1,
                convergence_tol_loss=1e-5,
                gt_path=gt_path,
                debug=True,
                use_rerun=False,
                early_stop="pose",
            )
            elapsed = time.time() - t0

            # Write mapping trajectory
            ts_synch_path = os.path.join(base_path, scene, gt_folder, "synch.txt")
            ts_opt_path   = f"./TUM_output/{scene}_seq_1_opt.txt"
            image_to_ts_map = image_to_ts(ts_synch_path)

            if os.path.exists(ts_opt_path):
                os.remove(ts_opt_path)
            adjuster.to_TUM(image_to_ts_map, ts_opt_path)

            # Evaluate mapping APE
            ts_gt_path = os.path.join(base_path, scene, gt_folder, "ts.txt")
            stats = eval_tum(ts_opt_path, ts_gt_path)
            print(f"[Mapping APE] scene={scene}  rmse={stats['rmse']:.4f}  mean={stats['mean']:.4f}  "
                  f"median={stats['median']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}  time={elapsed:.1f}s")
            
            os.system(
                    f"evo_ape tum {ts_gt_path} {ts_opt_path} -as -p --rerun"
                )

            # ----------------------------------------------------------------
            # Tracking phase: remaining frames from the same reconstruction
            # ----------------------------------------------------------------
            all_images      = sorted(adjuster.recon.images.values(), key=lambda x: x.name)
            tracking_frames = all_images[adjuster.n_mapping_frames:]

            ts_traq_path = f"./TUM_output/{scene}_seq_1_traq.txt"
            if os.path.exists(ts_traq_path):
                os.remove(ts_traq_path)

            frame_times = []

            for idx, image_t in tqdm(enumerate(tracking_frames), total=len(tracking_frames), desc=f"tracking {scene}"):
                t_frame = time.time()

                adjuster.add_frame(
                    images_path,
                    depths_path,
                    image_t.name,
                    image_t.camera,
                    image_t,
                    window_size=10,
                )
                adjuster.forward_frame(image_t.name, refinement_iteration=100)

                frame_times.append(time.time() - t_frame)

                # if idx % args.opt_interval == 0 and idx != 0:
                #     adjuster(
                #         window_pose=25,
                #         window_depth=50,
                #         window_loss=100,
                #         convergence_tol_pose=0.5,
                #         convergence_tol_depth=0.1,
                #         convergence_tol_loss=1e-5,
                #         debug=True,
                #         use_rerun=False,
                #         early_stop="none",
                #         refinement=True,
                #         refinement_iteration=1000,
                #     )

                adjuster.to_TUM(image_to_ts_map, ts_traq_path)

            if frame_times:
                print(f"Average tracking time per frame: {np.mean(frame_times)*1000:.1f} ms")

            # Evaluate tracking APE (if any tracking frames were processed)
            tracking_stats = None
            if tracking_frames and os.path.exists(ts_traq_path):
                tracking_stats = eval_tum(ts_traq_path, ts_gt_path)
                print(f"[Tracking APE] scene={scene}  rmse={tracking_stats['rmse']:.4f}  "
                      f"mean={tracking_stats['mean']:.4f}  median={tracking_stats['median']:.4f}  "
                      f"min={tracking_stats['min']:.4f}  max={tracking_stats['max']:.4f}")
                
                os.system(
                    f"evo_ape tum {ts_gt_path} {ts_traq_path} -as -p --rerun"
                )

            # ----------------------------------------------------------------
            # Collect scene results
            # ----------------------------------------------------------------
            scene_rmse.append(stats["rmse"])
            scene_time.append(elapsed)
            scenes_data[dataset][scene] = {
                "mapping": {
                    "rmse":   round(stats["rmse"],   4),
                    "mean":   round(stats["mean"],   4),
                    "median": round(stats["median"], 4),
                    "min":    round(stats["min"],    4),
                    "max":    round(stats["max"],    4),
                    "time":   round(elapsed,         2),
                },
            }
            if tracking_stats is not None:
                scenes_data[dataset][scene]["tracking"] = {
                    "rmse":   round(tracking_stats["rmse"],   4),
                    "mean":   round(tracking_stats["mean"],   4),
                    "median": round(tracking_stats["median"], 4),
                    "min":    round(tracking_stats["min"],    4),
                    "max":    round(tracking_stats["max"],    4),
                }

        ape_per_dataset.append(round(sum(scene_rmse) / len(scene_rmse), 4) if scene_rmse else 0.0)
        time_per_dataset.append(round(sum(scene_time) / len(scene_time), 2) if scene_time else 0.0)

    color = f"#{random.randint(0, 0xFFFFFF):06x}"
    entry = {
        "method": "EPO",
        "color": color,
        "ape_scores": ape_per_dataset,
        "time_scores": time_per_dataset,
        "scenes": scenes_data,
    }

    if os.path.exists(args.log) and os.path.getsize(args.log) > 0:
        with open(args.log) as f:
            log = json.load(f)
    else:
        log = {"metadata": {"datasets": datasets}, "experiments": [], "not_used": []}

    existing = next((e for e in log["experiments"] if e["method"] == "EPO"), None)
    if existing:
        existing.update(entry)
    else:
        log["experiments"].append(entry)

    with open(args.log, "w") as f:
        json.dump(log, f, indent=4)
    print(f"Results saved to {args.log}")
