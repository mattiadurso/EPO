import os
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import pycolmap
import warnings
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from itertools import combinations
import glob
from PIL import Image
from tqdm import tqdm
import rerun as rr

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Ignore the cuDNN warning
warnings.filterwarnings(
    "ignore",
    message=".*cudnnException.*CUDNN_STATUS_NOT_SUPPORTED.*",
)

# Ignore the tqdm / IProgress warning
warnings.filterwarnings(
    "ignore",
    message=".*IProgress not found.*",
)

from helpers.load import (
    find_images,
    process_pose,
    process_camera,
    load_and_preprocess_images,
    load_and_preprocess_depths,
)
from losses.dt_loss import (
    compute_distance_field_cv2,
    compute_chunk_loss_logic,
)
from helpers.reprojection import (
    filter_viewgraph_by_reprojection_batched,
    grid_sample_nan,
)
from helpers.reprojection_compiled import (
    unproject_2D_to_world,
    project_and_sample_logic,
)
from helpers.frustum import build_view_graph_from_frustums
from modules import *
from modules.stopping_criterion import evaluate_pose_changes
from epo_modules import *

import sys

sys.path.append("/home/mattia/Desktop/Repos/posebench/benchmarks_3D")
from benchmark_pose import eval_colmap_model


class EPO(nn.Module, MiscModule, ReconstructAndVizModule):
    """
    Module to adjust poses and intrinsics of a given reconstruction using edge alignment losses.
    Args:
        reconstruction_path (str): Path to the COLMAP reconstruction folder.
        images_path (str): Path to the folder containing input images.
        depths_path (str): Path to the folder containing depth maps.
        viewgraph_path (str, optional): Path to a precomputed viewgraph file. If None, it will be computed from frustums.
        sky_mask_path (str, optional): Path to the folder containing sky masks. Default is None.
            This excludes edges from sky regions. Might lead to less constraints in optimization and slightly worse results.
            Use when sky changes a lot between images. PRovide massk as png images with 1 for sky and 0 for non-sky.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
        single_camera_per_folder (bool, optional): Whether to assume a single camera per folder. Default is True.
        load_with_pad (bool, optional): Whether to load images with padding to make them square. Default is True.
        detector (str, optional): Edge detector to use. Default is "canny".
        device (str, optional): Device to use for computation. Default is "cuda".
        max_workers (int, optional): Maximum number of workers for parallel loading. Default is -1 (all available).
        detector_params (dict, optional): Parameters for the edge detector.
        seed (int, optional): Random seed for reproducibility. Default is 0.
        scheduler_name (str, optional): Learning rate scheduler name. Default is None.
        scheduler_params (dict, optional): Parameters for the learning rate scheduler.
        grad_q (bool, optional): Whether to optimize rotation quaternions. Default is True.
        grad_t (bool, optional): Whether to optimize translation vectors. Default is True.
        grad_k (bool, optional): Whether to optimize camera intrinsics. Default is True.
        grad_z (bool, optional): Whether to optimize depth scale. Default is False.
        gt_path (str, optional): Path to ground truth data for evaluation. Default is None.
        max_edges_points (int, optional): Maximum number of edge points to use per image. Default is 16384.
        max_viewgraph_pairs (int, optional): Maximum number of viewgraph pairs to use at each optimization step.
            It's randomly sampled from the full viewgraph at each iteration. Default is 8192. If pairs are greater than this, it's basically mini batch processing.
    """

    def __init__(
        self,
        reconstruction_path,
        images_path,
        depths_path,
        viewgraph_path=None,  # for testing with GT viewgraph
        unreliable_area_masks_path=None,
        images_size=518,
        single_camera_per_folder=True,
        load_with_pad=False,
        detector="canny",
        device="cuda",
        max_workers=-1,
        detector_params={
            "low_threshold": 0.20,
            "high_threshold": 0.25,
            "kernel_size": 7,
            "sigma": 2,
        },
        seed=0,
        max_edges_points=16_384,  # hard contraint due to memory on 24GB
        max_viewgraph_pairs=8_192,  # hard contraint due to memory on 24GB
        matcher_type="exhaustive",  # or "sequential"
        sequential_matcher_window=5,  # only for sequential matcher
        scene_type="outdoor",  # or "indoor", "object_centric" (not used yet)
        use_depth_confidence=False,
        q_lr=1e-4,
        t_lr=1e-3,
        k_lr=1e-3,
        z_lr=3e-3,
        mlp_pose_lr=3e-3,
        grad_q=False,
        grad_t=False,
        grad_t_offset=True,
        grad_k=True,
        grad_z=True,
        use_mlp_pose_refinement=True,
        auc_saving_freq=50,
        max_num_iterations=2000,
        verbose=False,
        # viewgraph params
        min_points=750,
        sampling_factor=5,
        reprojection_error=3,
        run_mode="inference",  # or "debug" for deterministic results and more logging
    ):
        super().__init__()
        self.device = device
        self.dtype = torch.float32

        # fix seed
        self.seed = seed
        self.mode = run_mode
        self.fix_seed(mode=run_mode)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)
        self.rng_cpu = torch.Generator(device="cpu")
        self.rng_cpu.manual_seed(self.seed)

        self.max_workers = os.cpu_count() if max_workers < 0 else max_workers
        self.images_size = images_size
        self.load_with_pad = load_with_pad
        self.images_path = images_path
        self.depths_path = depths_path
        self.reconstruction_path = reconstruction_path
        self.single_camera_per_folder = single_camera_per_folder
        self.sequential_matcher_window = sequential_matcher_window
        self.convergence = False
        self.auc_th = [1, 3, 5]
        self.completed_iterations = 0
        self.max_num_iterations = max_num_iterations
        self.auc_saving_freq = auc_saving_freq
        self.viewgraph_path = viewgraph_path
        self.matcher_type = matcher_type
        self.scene_type = scene_type
        self.max_edges = max_edges_points
        self.max_viewgraph_pairs = max_viewgraph_pairs
        self.unreliable_area_masks_path = unreliable_area_masks_path
        self.use_depth_confidence = use_depth_confidence
        self.verbose = verbose
        self.min_points = min_points
        self.sampling_factor = sampling_factor
        self.reprojection_error = reprojection_error

        # Edge extractor
        if detector == "canny":
            from extractors.canny import CannyEdgeDetector

            self.edge_extractor = CannyEdgeDetector(
                low_threshold=detector_params.get("low_threshold", 0.20),
                high_threshold=detector_params.get("high_threshold", 0.25),
                hysteresis=detector_params.get("hysteresis", True),
                kernel_size=detector_params.get("kernel_size", 7),
                sigma=detector_params.get("sigma", 2.0),
                device=device,
                verbose=verbose,
            )
        elif detector == "sam2":
            from extractors.SAM2.sam2_wrapper import SAM2EdgePointExtractor

            self.edge_extractor = SAM2EdgePointExtractor(device=device, size="large")
        elif detector == "bdcn":
            from extractors.BDCN.bdcn_wrapper import BDCNEdgeDetector

            self.edge_extractor = BDCNEdgeDetector(device=device)
        elif detector == "teed":
            from extractors.TEED.teed_wrapper import TeedWrapper

            self.edge_extractor = TeedWrapper(
                device=device,
            )
        elif detector == "diff":
            from extractors.DiffusionEdge.diffusion_edge_wrapper import (
                DiffusionEdgeDetector,
            )

            self.edge_extractor = DiffusionEdgeDetector(
                device=device,
            )
        elif detector == "rcf":
            from extractors.rcf_torch.rfc_wrapper import RCFWrapper

            self.edge_extractor = RCFWrapper(
                device=device,
            )
        else:
            raise ValueError(f"Unknown detector: {detector}")

        # what to train
        self.q_lr = q_lr
        self.t_lr = t_lr
        self.k_lr = k_lr
        self.z_lr = z_lr
        self.mlp_pose_lr = mlp_pose_lr
        self.grad_q = grad_q
        self.grad_t = grad_t
        self.grad_t_offset = grad_t_offset
        self.grad_k = grad_k
        self.grad_z = grad_z
        self.use_mlp_pose_refinement = use_mlp_pose_refinement

        # Loading
        self.timings = {}
        time_start = time.time()

        ## Load Reconstruction
        self.recon = pycolmap.Reconstruction(self.reconstruction_path)

        ## Load Images as dict {image_name: image_tensor}
        s_time = time.time()
        self.image_path_list = find_images(
            self.images_path
        )  # image name includes subfolder if any
        # loads image, coords, scale, hw into self.images[image_name]
        self._load_and_preprocess_images()
        self.num_images = len(self.images)
        self.timings["load_images"] = time.time() - s_time

        ## Load depth maps
        s_time = time.time()
        self._load_and_preprocess_depths()
        self.timings["load_depth_maps"] = time.time() - s_time

        ## Load poses and intrinsics
        s_time = time.time()
        # creating poses such self.poses[image_name] = PoseModel(...)
        # creating intrinsics such self.intrinsics[cam_id] = CameraModel(...)
        # using image name and camera id/folder str
        self._read_cameras_from_reconstruction()  # into self.images and self.intrinsics
        self.timings["load_poses_and_intrinsics"] = time.time() - s_time

        ## Extract edges
        s_time = time.time()
        self._extract_edges()  # into self.images
        self.timings["extract_edges"] = time.time() - s_time

        ## Compute Distance Fields
        s_time = time.time()
        self._compute_distance_fields()  # into self.images
        self.timings["compute_distance_fields"] = time.time() - s_time

        ## Viewgraph from frustums
        s_time = time.time()
        # compute viewgraph
        self._compute_viewgraph(
            type=self.matcher_type,
            min_points=min_points,
            sampling_factor=sampling_factor,
            reprojection_error=reprojection_error,
        )
        self.timings["compute_viewgraph"] = time.time() - s_time

        ## Prepare batched parameters modules that do not need to be optimized
        self.image_id_map = {}
        edges_padded, pad_masks = [], []
        dt_fields, images_shapes = [], []
        sampled_depth = []
        for idx, image_name in enumerate(sorted(self.images.keys())):
            # mapping image name to tensor index
            self.image_id_map[image_name] = idx
            # collecting data into big tensors
            edges_padded.append(self.images[image_name]["edges_padded"])
            pad_masks.append(self.images[image_name]["pad_mask"])
            dt_fields.append(self.images[image_name]["dt_field"])
            images_shapes.append(torch.tensor(self.images[image_name]["hw"]))
            sampled_depth.append(self.images[image_name]["sampled_depth"])

        # stacking
        edges_padded = torch.stack(edges_padded, dim=0).to(
            self.device, dtype=self.dtype
        )
        pad_masks = torch.stack(pad_masks, dim=0).to(self.device).bool()
        dt_fields = (
            torch.stack(dt_fields, dim=0).to(self.device, dtype=self.dtype).unsqueeze(1)
        )
        images_shapes = torch.stack(images_shapes, dim=0).to(
            self.device, dtype=self.dtype
        )
        sampled_depth = torch.stack(sampled_depth, dim=0).to(
            self.device, dtype=self.dtype
        )

        # storing
        self.edges_padded = BaseModule(self.image_id_map, edges_padded, self.device)
        self.pad_masks = BaseModule(self.image_id_map, pad_masks, self.device)
        self.dt_fields = BaseModule(self.image_id_map, dt_fields, self.device)
        self.images_hw = BaseModule(self.image_id_map, images_shapes, self.device)
        self.sampled_depth = DepthModule(
            image_id_map=self.image_id_map,
            depth=sampled_depth,
            device=self.device,
            lr=self.z_lr,
            grad=self.grad_z,
            max_num_iterations=self.max_num_iterations,
            dtype=self.dtype,
        )

        # Prepare viewgraph with indices for faster access during optimization
        viewgraph_ids = [
            (
                self.image_id_map[i],
                self.image_id_map[j],
                self.intrinsics.map_names_to_indices(self.images[i]["cam_id"]),
                self.intrinsics.map_names_to_indices(self.images[j]["cam_id"]),
            )
            for i, j in self.viewgraph
        ]

        # Also prepare image to cam id mapping tensor
        images_cams_ids = [
            (
                self.image_id_map[image_name],
                self.intrinsics.map_names_to_indices(self.images[image_name]["cam_id"]),
            )
            for image_name in sorted(self.images.keys())
        ]

        # (img1_id, img2_id, cam1_id, cam2_id)
        self.viewgraph_ids = torch.tensor(viewgraph_ids).long().to(self.device)
        # (image_id, cam_id)
        self.images_cams_ids = torch.tensor(images_cams_ids).long().to(self.device)

        # ==========================================================================
        # Create optimizer
        if self.verbose:
            params_to_optimize = self._collect_parameters_to_optimize()
            self._print_params_summary(params_to_optimize)

        self.timings["total_loading"] = time.time() - time_start
        self.timings["total_optimization"] = 0
        self.timings["step_pre_computation"] = 0
        self.timings["prepare_batched_inputs"] = 0
        self.timings["forward_pass"] = 0
        self.timings["loss_computation"] = 0
        self.timings["gradients_computation"] = 0
        self.timings["parameters_update"] = 0
        self.timings["logging"] = 0
        self.timings["early_stop_check"] = 0

        # # At this point I might get rid of rgb images to save memory as not needed for edge loss
        # for image_name in self.images.keys():
        #     self.images[image_name].pop("image")
        #     self.images[image_name].pop("depth")
        #     self.images[image_name].pop("edges_map")

        self.loss_list = []
        self.lr_list = {"q": [], "t": [], "mlp": [], "k": [], "z": []}
        self.auc_list = {"auc": {th: [] for th in self.auc_th}, "steps": []}
        self.convergence = False
        self.blacklist = set()
        self.changes = {"q": [], "t": [], "max": [], "steps": [], "z": []}
        self.mlp_pose_convergence = False
        self.optim_convergence = False
        self.convergence_loss = False

        gc.collect()
        torch.cuda.empty_cache()

    def forward(
        self,
        batch_size=256,
        residuals_chunk_size=2048,
        quantile=0.95,
        window_pose=25,
        window_depth=50,
        window_loss=100,
        convergence_tol_pose=0.5,  # degrees
        convergence_tol_depth=0.1,  # relative change %
        convergence_tol_loss=5e-5,  # relative change %
        early_stop="pose",
        drop_last=False,
        debug=False,
        gt_path=None,
        ba_path=None,
        use_rerun=False,
        spawn_rerun=True,
        rerun_save_path=".",
        scene_name="data",
        opt="optimized_reconstruction_GD/_current_test",
    ):
        """
        Main optimization loop.
        Args:
            batch_size (int, optional): Number of viewgraph pairs to process per batch. Default is 256.
            residuals_chunk_size (int, optional): Chunk size for residual computation. Default is 2048.
            quantile (float, optional): Quantile for evaluating pose changes. Default is 0.95.
            window_pose (int, optional): Window size for pose convergence evaluation. Default is 25.
            window_depth (int, optional): Window size for depth convergence evaluation. Default is 25.
            convergence_tol_pose (float, optional): Tolerance for pose convergence. Default is 0.5.
            convergence_tol_depth (float, optional): Tolerance for depth convergence. Default is 0.1. Not used when early_stop is False.
            early_stop (str, optional): Whether to stop early if depth convergence is reached. Default is 'pose'.
            recompute_viewgraph (bool, optional): Whether to recompute the viewgraph after first convergence. Default is False.
            drop_last (bool, optional): Whether to drop the last batch if smaller than batch_size. Default is False.
            debug (bool, optional): Whether to enable debug mode. Default is False.
            gt_path (str, optional): Path to the ground truth data. Default is None.
            ba_path (str, optional): Path to the bundle adjustment data. Default is None.
            use_rerun (bool, optional): Whether to use Rerun for visualization. Default is False.
            spawn_rerun (bool, optional): Whether to spawn a new Rerun instance. Default is True.
            rerun_save_path (str, optional): Path to save Rerun logs. Default is ".".
            scene_name (str, optional): Name of the scene. Default is "data".
            opt (str, optional): Path to the optimization output. Default is "optimized_reconstruction_GD/_current_test".

        """
        assert early_stop in ["none", "pose", "loss"]

        # assuming to to do not changes these or move to init
        self.window_pose = window_pose
        self.window_depth = window_depth
        self.window_loss = window_loss
        self.convergence_tol_pose = convergence_tol_pose
        self.convergence_tol_depth = convergence_tol_depth
        self.convergence_tol_loss = convergence_tol_loss
        self.optim_convergence = True if early_stop == "none" else False

        time_start = time.time()

        if use_rerun:
            rr.init("Feature-Less Optimization", spawn=spawn_rerun)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

            # Log Ground Truth if available
            if gt_path is not None:
                try:
                    print(f"Loading GT from {gt_path} for visualization...")
                    self.log_reconstruction_rerun(
                        gt_path,
                        entity="gt",
                        static_cameras=True,
                        points3D=True,
                        static_points=True,
                        camera_color=[48, 125, 73],
                    )
                except Exception as e:
                    print(f"Warning: Failed to load GT for Rerun visualization: {e}")
            if ba_path is not None:
                try:
                    print(f"Loading BA result from {ba_path} for visualization...")
                    self.log_reconstruction_rerun(
                        ba_path,
                        entity="ba",
                        static_cameras=True,
                        points3D=False,
                        static_points=True,
                        camera_color=[0, 50, 106],
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to load BA result for Rerun visualization: {e}"
                    )

        if self.verbose:
            num_batches = math.ceil(self.len_viewgraph / batch_size)
            num_batches = num_batches if drop_last else num_batches + 1
            total_points = self.max_edges * self.len_viewgraph
            print(
                f"Processing {self.len_viewgraph:,} pairs with batch size {batch_size:,} ({num_batches} batches per iteration).",
                f"Using {self.images[list(self.images.keys())[0]]['edges_padded'].numel()//2:,} edges per image.",  # // due to x and y
                "\n",
                f"Total points to process per iteration: {total_points:,}.",
                end="\n\n",
            )

        # store past poses for convergence evaluation
        past_poses = self.poses.get_all_matrices().detach().clone()
        self.timings["logging"] += time.time() - time_start

        # Forward and backward loop
        bar = tqdm(
            range(self.completed_iterations, self.max_num_iterations),
            total=self.max_num_iterations,
            initial=self.completed_iterations,
            desc="Optimizing the scene",
        )
        for step in bar:
            t_pre = time.time()

            # Initialize optimizer gradients for all optimizers
            self.optimizers_zero_grad()

            # Update geometric modules
            self.poses.update_all_matrices()
            self.intrinsics.update_all_matrices()

            # Unproject point to world coordinates
            self.unproject_edges_to_3D()

            self.timings["step_pre_computation"] += time.time() - t_pre

            # Compute residuals
            residuals, sampled_viewgraphs = self.compute_forward_step(
                self.viewgraph_ids,
                batch_size=batch_size,
                drop_last=drop_last,
                residuals_chunk_size=residuals_chunk_size,
            )

            # Compute loss
            loss = self.compute_batched_loss(residuals, sampled_viewgraphs, debug=debug)
            s_time = time.time()
            loss.backward()
            self.timings["gradients_computation"] += time.time() - s_time

            self.optimizer_and_scheduler_step()

            if use_rerun:
                # Handle Rerun API variations
                if hasattr(rr, "set_time_sequence"):
                    rr.set_time_sequence("step", step)

                self.to_colmap(
                    opt,
                    verbose=False,
                    max_points_per_image=100_000 // self.num_images,
                    save_points=False,
                    final_dbscan_filtering=False,
                    dbscan_eps=0.1,
                    dbscan_min_samples=5,
                    gt_path=gt_path,  # to align
                )

                self.log_reconstruction_rerun(
                    opt,
                    entity="opt",
                    static_cameras=False,
                    points3D=False,
                    static_points=False,
                    camera_color=[186, 39, 34],  # red
                )

            # ============================================================
            # Logging
            # ============================================================
            logging_time_start = time.time()
            self.loss_list.append(loss.detach().item())
            self.collect_lrs(len(self.loss_list) - 1)

            # Evaluate AUC if GT available
            if gt_path is not None and step % self.auc_saving_freq == 0:
                self.to_colmap(opt, save_points=False, verbose=False)
                self.compute_auc(opt, gt_path, step)

            if self.verbose:
                bar.set_postfix(
                    loss=f"{self.loss_list[-1]:.4f}",
                    auc5=(
                        f"{self.auc_list['auc'][5][-1]:.4f}"
                        if len(self.auc_list["auc"][5]) > 0
                        else 0
                    ),
                )

            # rerun tracking
            if use_rerun:
                if gt_path is not None and len(self.auc_list["auc"][1]) > 0:
                    for th in self.auc_th:
                        rr.log(
                            f"metrics/AUC@{th}",
                            rr.Scalars(self.auc_list["auc"][th][-1]),
                        )

            self.timings["logging"] += time.time() - logging_time_start
            # ============================================================
            # Early stopping
            # ============================================================
            early_stop_start = time.time()

            # collect pose changes for convergence evaluation
            current_poses = self.poses.get_all_matrices().detach().clone()
            err_q, err_t, max_err = evaluate_pose_changes(
                past_poses,
                current_poses,
                quantile=quantile,
            )
            if self.verbose:
                self.changes["q"].append(err_q)
                self.changes["t"].append(err_t)
            self.changes["max"].append(max_err)
            self.changes["steps"].append(step)
            past_poses = current_poses

            if not self.mlp_pose_convergence:
                mlp_pose_convergence = self.check_convergence(
                    list_of_changes=self.changes["max"],
                    window=window_pose,
                    early_stop="pose",
                    tol=convergence_tol_pose,
                )
                if mlp_pose_convergence:
                    self.mlp_pose_convergence = True
                    self.timings["pose_convergence_time"] = time.time() - time_start
                    if self.verbose:
                        print(f"Pose convergence reached at step {step}.")

                    # If not optimizing depth, mark it as converged immediately
                    if not self.grad_z:
                        self.optim_convergence = True
                        self.timings["depth_convergence_time"] = (
                            time.time() - time_start
                        )

            elif self.mlp_pose_convergence:
                if early_stop == "pose":
                    if self.check_convergence(
                        list_of_changes=self.changes["max"],
                        window=window_depth,
                        early_stop=early_stop,  # "pose"
                        tol=convergence_tol_depth,
                    ):
                        self.optim_convergence = True

                elif early_stop == "loss":
                    if self.check_convergence(
                        list_of_changes=self.loss_list,
                        window=window_loss,
                        early_stop=early_stop,  # "loss"
                        tol=convergence_tol_loss,
                    ):
                        self.optim_convergence = True

                elif early_stop == "none":
                    # nothing to do in this case
                    pass

            self.timings["early_stop_check"] += time.time() - early_stop_start

            if (
                self.mlp_pose_convergence
                and self.optim_convergence
                and early_stop != "none"
            ):
                if self.verbose:
                    print(f"Stopping optimization at step {step}. Convergence reached.")
                self.completed_iterations += 1
                break

            self.completed_iterations += 1

        self.timings["total_optimization"] += time.time() - time_start

        if use_rerun:
            rr_folder = os.path.join(rerun_save_path, "rerun")
            os.makedirs(rr_folder, exist_ok=True)
            rr.save(os.path.join(rr_folder, f"{scene_name}.rrd"))

        if gt_path is not None:
            self.to_colmap(opt, save_points=False, verbose=False)
            self.compute_auc(opt, gt_path, step)

        self.compute_mre()
        self.print_summary() if self.verbose else print("=" * 70, end="\n\n")

    ### Forward and backward helpers ###
    def check_convergence(self, list_of_changes, early_stop, window, tol):
        """
        Evaluate convergence based on pose changes: max(delta_r, delta_t).
        Stop when smoothed max change is below tol for 'window' consecutive steps.
        """
        # We need at least 2*window - 1 steps to have 'window' smoothed points
        # to check for stability over 'window' steps.
        required_len = int(2 * window - 1)

        if len(list_of_changes) < required_len:
            return False

        # Get the last chunk of data needed to compute the last 'window' smoothed values
        # We need 'window' smoothed values.
        # The last smoothed value uses pose_changes[-window:]
        # The first of the 'window' smoothed values uses pose_changes[-(2*window-1) : -(window-1)]
        # So we need the last 2*window - 1 raw values.
        recent_changes = list_of_changes[-required_len:]

        # Sanitize input: convert Tensors to float
        recent_changes = [x.item() if torch.is_tensor(x) else x for x in recent_changes]

        # Compute smoothed max changes for this chunk
        smoothed = np.convolve(recent_changes, np.ones(window) / window, mode="valid")

        if early_stop == "loss":
            # Relative change: (current - previous) / previous
            smoothed = np.abs(np.diff(smoothed) / (np.abs(smoothed[:-1]) + 1e-8))

        # Check if ALL of them are below tolerance.
        return np.all(smoothed < tol)

    def compute_auc(self, opt, gt_path, step):
        AUC_score_max, num_images, _ = eval_colmap_model(
            opt, gt_path, return_df=False, thrs=self.auc_th
        )
        # store AUC
        for i, th in enumerate(self.auc_th):
            self.auc_list["auc"][th].append(AUC_score_max[i].item())
        self.auc_list["steps"].append(step)

    def optimizers_zero_grad(self):
        """Zero the gradients of all optimizers."""

        if hasattr(self.intrinsics, "optimizer"):
            self.intrinsics.optimizer.zero_grad()

        if hasattr(self.poses, "optimizer"):
            self.poses.optimizer.zero_grad()

        if hasattr(self.sampled_depth, "optimizer"):
            self.sampled_depth.optimizer.zero_grad()

    def optimizer_and_scheduler_step(self):
        """Perform optimizer step and scheduler step for all optimizers."""
        # Ideally each of them should be able to run independently until needed (reaching min lr),
        # but for now we keep them in sync for simplicity.
        s_time = time.time()
        if (
            self.grad_q is True
            or self.grad_t is True
            or self.use_mlp_pose_refinement is True
        ):
            self.poses.optimizer_and_scheduler_step()

        if self.grad_z is True and self.mlp_pose_convergence is True:
            # backprop on this without first stabilizing the mlp leads to bad stuff
            self.sampled_depth.optimizer_and_scheduler_step()

        # independent from phase
        if self.grad_k:
            self.intrinsics.optimizer_and_scheduler_step()

        self.timings["parameters_update"] += time.time() - s_time

    def collect_lrs(self, step):
        """Collect learning rates for all optimizers."""

        if hasattr(self.poses, "scheduler"):
            # these two are mutually exclusive
            if self.use_mlp_pose_refinement:
                self.lr_list["mlp"].append(
                    (step, self.poses.scheduler.get_last_lr()[0])
                )
            else:
                # get for group with name 't' and 'q'
                for param_group in self.poses.optimizer.param_groups:
                    if param_group["name"] == "t":
                        self.lr_list["t"].append((step, param_group["lr"]))
                    elif param_group["name"] == "q":
                        self.lr_list["q"].append((step, param_group["lr"]))

        if hasattr(self.intrinsics, "scheduler"):
            self.lr_list["k"].append((step, self.intrinsics.scheduler.get_last_lr()[0]))

        if hasattr(self.sampled_depth, "scheduler"):
            self.lr_list["z"].append(
                (step, self.sampled_depth.scheduler.get_last_lr()[0])
            )

    def unproject_edges_to_3D(self, batch_size=None):
        """Unproject 2D edges to 3D points for all images as a batch."""
        image_names_id = self.images_cams_ids[:, 0]
        cam_ids = self.images_cams_ids[:, 1]

        # indexing data
        K_batch = self.intrinsics.get_intrinsic_matrix(cam_ids)  # (B, 3, 3)
        P_batch = self.poses.get_projection_matrix(image_names_id)  # (B, 4, 4)
        edges_batch = self.edges_padded.get_parameters(image_names_id)  # (B, N, 2)
        depth_batch = self.sampled_depth.get_parameters(image_names_id)  # (B, 1, H, W)

        # Optionally chunk if batch too large for memory
        B = len(K_batch)
        if batch_size is None:
            batch_size = B
        points_3D_list = []

        for i in range(0, B, batch_size):
            xy0 = edges_batch[i : i + batch_size]
            K0 = K_batch[i : i + batch_size]
            depth0 = depth_batch[i : i + batch_size]
            P0 = P_batch[i : i + batch_size]

            pts3d = unproject_2D_to_world(
                xy0=xy0, K0=K0, depth0=depth0, P0=P0
            )  # (bs, N, 3)
            points_3D_list.append(pts3d)

        points_3D = torch.cat(points_3D_list, dim=0)

        # Store points_3D in Edges3DModule
        if not hasattr(self, "edges_3D"):
            self.edges_3D = BaseModule(
                image_id_map=self.image_id_map,
                parameters=points_3D,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.edges_3D.params = points_3D

    def create_batched_inputs(self, sampled_viewgraph):
        """Prepare batched inputs for the batched optimization step given a list of pairs from the viewgraph."""
        # mantain str index for visualization/debugging
        # if isinstance(sampled_viewgraph[0][0], str):
        #     cam_ids = []
        #     images_names_ij = []
        #     images_names_ji = []
        #     for (
        #         i,
        #         j,
        #     ) in sampled_viewgraph:  # with i,j being left (0) and right (1) images
        #         # I already have unprojected points to 3D, so I only have to check that
        #         # those points reproject within the image boundaries.

        #         # these are are the ids for points 3D and their corresponding pad
        #         images_names_ij.append(i)
        #         images_names_ij.append(j)

        #         # these are the ids for right images where to reproject
        #         cam_ids.append(self.images[j]["cam_id"])
        #         cam_ids.append(self.images[i]["cam_id"])
        #         images_names_ji.append(j)
        #         images_names_ji.append(i)

        # else:
        # sampeld_viewgraph is tensor of shape (num_pairs, 4) with (img1_id, img2_id, cam1_id, cam2_id)
        images_names_ij = sampled_viewgraph[:, :2].reshape(-1)
        images_names_ji = sampled_viewgraph[:, :2].flip(1).reshape(-1)
        cam_ids = sampled_viewgraph[:, 2:].flip(1).reshape(-1)

        batch = {}
        # 3D points in world coordinates and padd for left images
        batch["xyz_world"] = self.edges_3D.get_parameters(images_names_ij)
        pad_masks = self.pad_masks.get_parameters(images_names_ij)

        # these are the intrinsics and poses for right images. Needed to project
        # 3D world points to the second image of the pair
        batch["K1"] = self.intrinsics.get_intrinsic_matrix(cam_ids)
        batch["P1"] = self.poses.get_projection_matrix(images_names_ji)
        batch["img1_shape"] = self.images_hw.get_parameters(images_names_ji[:1])
        dt_fields = self.dt_fields.get_parameters(images_names_ji)

        return batch, pad_masks, dt_fields

    def compute_forward_step(
        self,
        sampled_viewgraph,
        batch_size=1024,
        residuals_chunk_size=1024,
        drop_last=True,
    ):
        """Compute one optimization step over the sampled_viewgraph in a batched manner and return the loss."""
        # reduce viewgraph if too large
        if len(sampled_viewgraph) > self.max_viewgraph_pairs:
            # indices = torch.randperm(len(sampled_viewgraph))[: self.max_viewgraph_pairs]
            indices = torch.randperm(len(sampled_viewgraph), generator=self.rng_cpu)[
                : self.max_viewgraph_pairs
            ]
            sampled_viewgraph = sampled_viewgraph[indices]

        # divide self.viewgraph in batches if len(self.viewgraph) > batch size
        sampled_viewgraphs = []
        if len(sampled_viewgraph) > batch_size:
            for i in range(0, len(sampled_viewgraph), batch_size):
                end = min(i + batch_size, len(sampled_viewgraph))
                sampled_viewgraphs.append(sampled_viewgraph[i:end])
        else:
            sampled_viewgraphs.append(sampled_viewgraph)

        if (
            len(sampled_viewgraphs) > 1  # to avoid dropping when only one batch
            and len(sampled_viewgraphs[-1]) < batch_size
            and drop_last
        ):
            sampled_viewgraphs = sampled_viewgraphs[:-1]

        # collect per-batch results in a python list (tensors)
        residuals_list = []
        # i might want to process batches of same size and drop last batch
        for sampled_viewgraph in sampled_viewgraphs:
            # prepare batched inputs
            s_time = time.time()
            batch, pad_masks, dt_fields = self.create_batched_inputs(sampled_viewgraph)
            self.timings["prepare_batched_inputs"] += time.time() - s_time

            # actual inference
            s_time = time.time()

            # projection and sampling
            residuals, inside_mask = project_and_sample_logic(
                batch["xyz_world"],
                batch["K1"],
                batch["P1"],
                batch["img1_shape"],
                dt_fields,
                border=0,
            )

            # chunked computation of loss over residuals
            valid_mask = pad_masks & inside_mask

            B, N = residuals.shape
            total_sum = torch.zeros(B, device=self.device, dtype=self.dtype)
            total_count = torch.zeros(B, device=self.device, dtype=torch.long)

            for i in range(0, N, residuals_chunk_size):
                r_chunk = residuals[:, i : i + residuals_chunk_size]
                m_chunk = valid_mask[:, i : i + residuals_chunk_size]

                # Pass the clean residual chunk and the merged validity mask
                s_chunk, c_chunk = compute_chunk_loss_logic(r_chunk, m_chunk)

                total_sum += s_chunk
                total_count += c_chunk

            zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            mean_losses = torch.where(
                total_count > 0,
                total_sum / total_count.to(self.dtype).clamp(min=1.0),
                zero,
            )

            # collect this batch's results
            residuals_list.append(mean_losses)

            self.timings["forward_pass"] += time.time() - s_time

        # concatenate all collected batch results
        residuals = torch.cat(residuals_list, dim=0)  # (num_pairs,)

        return residuals, sampled_viewgraphs

    def compute_batched_loss(
        self, residuals, sampled_viewgraphs=None, debug=False, delta=1.0
    ):
        """Vectorized batched loss computation."""
        s_time = time.time()

        if residuals.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        residuals_pairs = residuals.view(-1, 2)
        residuals_pairs = torch.clamp(residuals_pairs, min=0, max=10.0)

        # Use Huber loss for robust cost
        pair_losses = F.huber_loss(
            residuals_pairs,
            torch.zeros_like(residuals_pairs),
            reduction="none",
            delta=delta,
        )
        pair_losses = pair_losses.sum(dim=1)  # (num_pairs,)

        # if sampled_viewgraphs is given, store per-pair losses for logging
        if sampled_viewgraphs is not None and debug:
            if not hasattr(self, "residuals"):
                self.residuals = []

            residuals_iteration = {}
            pair_idx = 0
            for viewgraph in sampled_viewgraphs:
                for i, j, _, _ in viewgraph:
                    residuals_iteration[(i, j)] = (
                        pair_losses[pair_idx].detach().cpu().item()
                    )
                    pair_idx += 1
            self.residuals.append(residuals_iteration)

        # Mean over pairs
        loss = pair_losses.mean()

        self.timings["loss_computation"] += time.time() - s_time

        return loss

    ### Helper functions for loading and preprocessing data ###
    def _read_cameras_from_reconstruction(self):
        intrinsics = {}

        # Read cameras intrinsics
        if self.single_camera_per_folder:
            # Reading cameras from images (to handle multiple images with same camera)
            for image in self.recon.images.values():
                _, model, new_params = process_camera(
                    image.camera, self.load_with_pad, images_size=self.images_size
                )
                # assuming image names are like "cam_id/image_name"
                cam_id = image.name.split("/")[0]

                # I want to stack params of same cam_id to then averaged them
                if cam_id not in intrinsics:
                    intrinsics[cam_id] = {
                        "cam_id": cam_id,
                        "model": model,
                        "parameters": [new_params],
                    }
                else:
                    # Append new params to the list
                    intrinsics[cam_id]["parameters"].append(new_params)

            # Average params for each cam_id
            for cam_id in intrinsics.keys():
                params = intrinsics[cam_id]["parameters"]
                if len(params) == 1:
                    # only one image with this cam_id
                    intrinsics[cam_id]["parameters"] = params[0]
                else:
                    # multiple images with this cam_id - stack and average
                    intrinsics[cam_id]["parameters"] = torch.stack(params, dim=0).mean(
                        dim=0
                    )
        else:  # one camera per image
            # Reading cameras from images
            for cam in self.recon.cameras.values():
                cam_id, model, new_params = process_camera(cam)
                intrinsics[cam.camera_id] = {
                    "cam_id": cam_id,
                    "model": model,
                    "parameters": new_params.to(self.device),
                }
        # Sort dict by keys
        intrinsics = dict(sorted(intrinsics.items()))

        # Convert to Camera objects
        cam_id_to_tensor_id = {}
        k_models, k_params = [], []
        for idx, cam_id in enumerate(sorted(intrinsics.keys())):
            cam_id_to_tensor_id[cam_id] = idx
            k_models.append(intrinsics[cam_id]["model"])
            k_params.append(intrinsics[cam_id]["parameters"])

        intrinsics = CameraModule(
            image_id_map=cam_id_to_tensor_id,
            k_models=k_models,
            k_params=torch.stack(k_params),
            lr=self.k_lr,
            device=self.device,
            dtype=self.dtype,
            grad=self.grad_k,
            max_num_iterations=self.max_num_iterations,
        )

        # Read poses from images
        poses_temp = {}
        for image in self.recon.images.values():
            R, t, cam_id = process_pose(image)

            # trusting the folders structure, VGGT returns one camera per image a priori
            if self.single_camera_per_folder:
                cam_id = image.name.split("/")[0]
            else:
                cam_id = image.camera_id

            poses_temp[image.name] = {"R": R, "t": t, "cam_id": cam_id}

        images_id_map = {}
        R_tensor = []
        t_tensor = []
        for idx, image_name in enumerate(sorted(poses_temp.keys())):
            images_id_map[image_name] = idx
            self.images[image_name]["cam_id"] = poses_temp[image_name]["cam_id"]
            R_tensor.append(poses_temp[image_name]["R"])
            t_tensor.append(poses_temp[image_name]["t"])

        poses = PoseModule(
            images_id_map,
            hw=self.images[image_name]["hw"],
            R=torch.stack(R_tensor),
            t=torch.stack(t_tensor),
            q_lr=self.q_lr,
            t_lr=self.t_lr,
            grad_q=self.grad_q,
            grad_t=self.grad_t,
            grad_t_offset=self.grad_t_offset,
            mlp_lr=self.mlp_pose_lr,
            use_mlp=self.use_mlp_pose_refinement,
            max_num_iterations=self.max_num_iterations,
            device=self.device,
            dtype=self.dtype,
        )

        self.poses = poses
        self.intrinsics = intrinsics

    def _load_and_preprocess_images(self):
        self.images = load_and_preprocess_images(
            self.image_path_list,
            self.images_path,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=self.load_with_pad,
            dtype=self.dtype,
            device=self.device,
        )

    def _load_and_preprocess_depths(self):
        self.images = load_and_preprocess_depths(
            self.depths_path,
            self.images,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=self.load_with_pad,
            dtype=self.dtype,
            device=self.device,
        )
        # check all depths have the same size
        depth_shapes = set()
        for image_name in self.images.keys():
            depth_shapes.add(self.images[image_name]["depth"].shape[-2:])
        if len(depth_shapes) > 1:
            # pad bottom right to make them equal
            max_h = max([shape[0] for shape in depth_shapes])
            max_w = max([shape[1] for shape in depth_shapes])
            for image_name in self.images.keys():
                depth = self.images[image_name]["depth"]
                h, w = depth.shape[-2:]
                if h < max_h or w < max_w:
                    pad_bottom = max_h - h
                    pad_right = max_w - w
                    pad = (0, pad_right, 0, pad_bottom)  # left, right, top, bottom
                    depth = F.pad(depth, pad, mode="constant", value=torch.nan)
                    self.images[image_name]["depth"] = depth.to(
                        self.device, dtype=self.dtype
                    )

    @torch.no_grad()
    def _compute_viewgraph(
        self, type="exhaustive", min_points=750, sampling_factor=5, reprojection_error=3
    ):
        """Compute viewgraph and filter by reprojection error and returns the sorted viewgraph."""
        if self.viewgraph_path is not None:
            # Load viewgraph from file
            with open(self.viewgraph_path, "r") as f:
                lines = f.readlines()
            viewgraph = []
            for line in lines:
                i, j, m = line.strip().split()
                viewgraph.append((i, j))
            self.viewgraph = viewgraph

        elif type == "frustums":
            # Estimate view graph from frustums
            viewgraph = build_view_graph_from_frustums(
                self.recon,
                max_view_angle_deg=30.0,
                distance_factor=2,
                verbose=False,
                images_with_depth=self.images,
                dtype=self.dtype,
            )
            # min_points = 100
            # sampling_factor = 5
            # reprojection_error = 5.0

        elif type == "sequential":
            # Build sequential viewgraph based on sorted image names with a window size of 10
            image_names = sorted(list(self.images.keys()))
            viewgraph = []
            for i in range(len(image_names) - self.sequential_matcher_window):
                for j in range(1, self.sequential_matcher_window + 1):
                    viewgraph.append((image_names[i], image_names[i + j]))
            min_points = 200
            sampling_factor = 5
            reprojection_error = 3.0

        elif type == "exhaustive":
            # Build exhaustive viewgraph (all pairs)
            image_names = sorted(list(self.images.keys()))
            viewgraph = list(combinations(image_names, 2))
            # min_points = 750  # ~12.5% (1/8) of 518*290
            # sampling_factor = 5  # after sampling
            # reprojection_error = 3.0

        else:
            raise ValueError(f"Viewgraph type {type} not supported.")

        # Filter viewgraph by reprojection
        # min_points shouldbe set such images have enough overlap, maybe at least 25%
        # 518*345*0.25 = 44_677
        if self.viewgraph_path is None:
            self.viewgraph, self.valid_points_per_pair = (
                filter_viewgraph_by_reprojection_batched(
                    viewgraph=viewgraph,
                    images=self.images,
                    intrinsics=self.intrinsics,
                    poses=self.poses,
                    # === parameters =====
                    min_points=min_points,
                    sampling_factor=sampling_factor,
                    reprojection_error=reprojection_error,
                    border=10,
                    # ====================
                    device=self.device,
                    verbose=self.verbose,
                )
            )

        self.viewgraph.sort(key=lambda x: (x[0], x[1]))

        self.len_viewgraph = len(self.viewgraph)
        self.adj_list = {}
        for i, j in self.viewgraph:
            self.adj_list.setdefault(i, []).append(j)
            self.adj_list.setdefault(j, []).append(i)

        if self.verbose:
            values = [len(v) for v in self.adj_list.values()]
            if len(values) > 0:
                print(
                    f"Average degree: {np.mean(values):.2f}, min connection {min(values)}, less than 5 neighbors: {(np.array(values)<5).sum()} images"
                )
            else:
                print("\n >>> No valid pairs in viewgraph, which is a BIG problem! <<<")

    ### Edges
    def _extract_edges(self, confidence_threshold=0.2):
        tot_edges = 0
        # for image_name in tqdm(self.images.keys(), desc=f"Extracting edges"):
        for image_name in self.images.keys():
            # Extract edge map
            img_tensor = self.images[image_name]["image"].unsqueeze(0)
            edges_map = (
                self.edge_extractor(img_tensor)
                .squeeze()
                .to(self.device, dtype=self.dtype)
            )

            # Discard points at invalid depth locations
            if "confidence" in self.images[image_name] and self.use_depth_confidence:
                confidence = self.images[image_name]["confidence"]
                # create valid depth mask, normalize between 0 and 1
                confidence = (confidence - confidence.min()) / (
                    confidence.max() - confidence.min()
                )
                valid_depth_mask = (confidence > confidence_threshold) & (
                    ~torch.isnan(confidence)
                )
                valid_edges_map = edges_map * valid_depth_mask.to(self.dtype)

                # Extract edges from valid edges map
                valid_edges = (
                    valid_edges_map.squeeze().nonzero().flip(dims=(1, 0))
                )  # (N, 2)
            else:
                valid_edges_map = edges_map
                # Extract edges from edges map
                valid_edges = edges_map.squeeze().nonzero().flip(dims=(1, 0))  # (N, 2)

            # Store edges and edges map
            self.images[image_name]["edges_map"] = valid_edges_map
            self.images[image_name]["edges"] = valid_edges.to(
                self.device, dtype=self.dtype
            )
            tot_edges += valid_edges.shape[0]
        self.observations = tot_edges

        # pad to have same number of edges per image
        self._pad_edges()

        # add sampled depth at edges_padded locations
        for image_name in self.images.keys():
            edges_padded = self.images[image_name]["edges_padded"]  # (N, 2)
            depth = self.images[image_name]["depth"]  # (H, W)
            sampled_depth, _ = grid_sample_nan(edges_padded[None], depth[None])
            # if invalid points set at 0,0 have nan depth, then I'll have sampled depth as nan.
            # fill with zeros, these points will be masked out during optimization anyway
            sampled_depth = torch.where(
                torch.isnan(sampled_depth),
                torch.zeros_like(sampled_depth) + 1e-6,
                sampled_depth,
            )
            self.images[image_name]["sampled_depth"] = sampled_depth.squeeze()

    def _pad_edges(self):
        """Pad all edges to have same number (max_edges) of edges per image."""
        num_edges = [self.images[img]["edges"].shape[0] for img in self.images.keys()]
        max_edges = max(num_edges)
        min_edges = min(num_edges)
        std_edges = torch.std(torch.tensor(num_edges, dtype=self.dtype)).item()
        avg_edges = sum(num_edges) / len(num_edges)
        median_edges = sorted(num_edges)[len(num_edges) // 2]
        q90 = (
            torch.quantile(torch.tensor(num_edges, dtype=torch.float32), 0.9)
            .long()
            .item()
        )

        # this to save some computation/memory
        # likely only few images have very large number of edges
        max_edges_to_retain = min(self.max_edges, min(q90, max_edges))
        images_with_more_than_max = sum(1 for n in num_edges if n > max_edges_to_retain)

        for image_name in self.images.keys():
            edges = self.images[image_name]["edges"]
            n_edges = edges.shape[0]

            if n_edges > max_edges_to_retain:
                # randomly sample max_edges
                # indices = torch.randperm(n_edges, device=edges.device)[
                #     :max_edges_to_retain
                # ]
                indices = torch.randperm(
                    n_edges, device=edges.device, generator=self.rng
                )[:max_edges_to_retain]

                edges = edges[indices]
                n_edges = max_edges_to_retain

            if n_edges < max_edges_to_retain:
                pad_size = max_edges_to_retain - n_edges
                pad = torch.zeros((pad_size, 2), device=edges.device)
                edges = torch.cat([edges, pad], dim=0).to(self.dtype)

                pad_mask = torch.zeros(
                    (max_edges_to_retain,), device=edges.device, dtype=self.dtype
                )
                pad_mask[:n_edges] = 1.0
            else:
                # n_edges == max_edges_to_retain: all edges are valid
                pad_mask = torch.ones(
                    (max_edges_to_retain,), device=edges.device, dtype=self.dtype
                )

            self.images[image_name].update(
                {"edges_padded": edges, "pad_mask": pad_mask}
            )

        self.max_edges = max_edges_to_retain

        if self.verbose:
            print(
                f"Edges stats:\n",
                f"{images_with_more_than_max:,} images have more than {max_edges_to_retain:,} edges. \n",
                f"max: {max_edges:,} |",
                f"min: {min_edges:,} | avg: {int(avg_edges):,} |",
                f"std: {std_edges:,.2f} |",
                f"quantiles (0.5, 0.9): {median_edges:,}, {q90:,}",
            )

    @torch.no_grad()
    def _compute_distance_fields(self):
        dt_fields_shapes = []
        # for image_name in tqdm(self.images.keys(), desc="Computing distance fields"):
        for image_name in self.images.keys():
            edges_map = self.images[image_name]["edges_map"]
            dt_field = compute_distance_field_cv2(
                edges_map,
                device=self.device,
            )
            self.images[image_name].update(
                {"dt_field": dt_field.to(self.device, dtype=self.dtype)}
            )
            dt_fields_shapes.append(dt_field.shape)

        # if dt_fields_shapes is not equal, need to pad right bottom to make them equal
        if len(set(dt_fields_shapes)) > 1:
            max_h = max([shape[0] for shape in dt_fields_shapes])
            max_w = max([shape[1] for shape in dt_fields_shapes])
            for image_name in self.images.keys():
                dt_field = self.images[image_name]["dt_field"]
                h, w = dt_field.shape
                if h < max_h or w < max_w:
                    pad_bottom = max_h - h
                    pad_right = max_w - w
                    pad = (0, pad_right, 0, pad_bottom)  # left, right, top, bottom
                    dt_field = F.pad(
                        dt_field, pad, mode="constant", value=dt_field.max()
                    )
                    self.images[image_name]["dt_field"] = dt_field

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_mre(self):
        """Measuring how far each image_i's edges are projected to the closest edges in image_j"""
        # Update geometric modules
        self.poses.update_all_matrices()
        self.intrinsics.update_all_matrices()

        # Unproject point to world coordinates
        self.unproject_edges_to_3D()

        # Compute residuals
        residuals, sampled_viewgraphs = self.compute_forward_step(
            self.viewgraph_ids,
            batch_size=10_000,
            drop_last=False,
            residuals_chunk_size=10_000,
        )

        sampled_viewgraphs = sampled_viewgraphs[0][:, :2].tolist()

        for _ in range(len(sampled_viewgraphs)):
            i, j = sampled_viewgraphs[_]
            sampled_viewgraphs[_] = [
                self.poses.tensor_idx_to_image[i],
                self.poses.tensor_idx_to_image[j],
                residuals[_].item(),
            ]
        sampled_viewgraphs = np.array(sampled_viewgraphs)
        sampled_viewgraphs[:, 2] = sampled_viewgraphs[:, 2].astype(np.float32)

        self.mre = sampled_viewgraphs[:, 2].astype(np.float32)

        return sampled_viewgraphs


if __name__ == "__main__":
    import json

    dataset = "scannetpp"  # terrasky3D, mipnerf360,
    scene = "5f99900f09"  # vienna_state_opera, bicycle, bonsai, statue, 7831862f02
    model = "vggt"

    # Load dataset paths and parameters from JSON
    with open("benchmarks/paths.json") as f:
        paths_cfg = json.load(f)

    # for dataset in ["terrasky3D", "scannetpp"]:
    #     scenes = sorted(os.listdir(paths_cfg[dataset]["images_path"]))

    #     for scene in scenes:
    # if os.path.exists(f"rerun/{scene}.rrd") or scene[0] == "." or scene.endswith(".py"):
    #     continue

    dataset_cfg = paths_cfg[dataset]

    images_path = os.path.join(
        dataset_cfg["images_path"], scene, dataset_cfg["images_folder"]
    )
    base_path = dataset_cfg["base_path"]
    reconstruction_path = os.path.join(
        base_path, scene, dataset_cfg["reconstruction_folder"]
    )
    depths_path = os.path.join(
        base_path,
        scene,
        dataset_cfg.get("depths_folder", dataset_cfg.get("depth_folder", "")),
    )
    gt_path = os.path.join(dataset_cfg["gt_path"], scene, dataset_cfg["gt_folder"])

    reconstruction_path = reconstruction_path.replace("vggt", model)
    depths_path = depths_path.replace("vggt", model)

    epo = EPO(
        reconstruction_path=reconstruction_path,
        images_path=images_path,
        depths_path=depths_path,
    )

    epo(
        batch_size=256,
        residuals_chunk_size=2048,
        window_pose=25,
        window_depth=50,
        gt_path=gt_path,
        ba_path=reconstruction_path.replace("vggt", "vggt_ba_ref"),
        use_rerun=False,
        spawn_rerun=False,
        rerun_save_path=".",
        scene_name=scene,
    )

    # Saving
    save_points = True  # recall to set mean track len = 0 in colmap gui
    opt = "optimized_reconstruction_GD/_current_test"

    epo.to_colmap(
        opt,
        verbose=False,
        max_points_per_image=100_000 // epo.num_images,
        save_points=save_points,
        final_dbscan_filtering=False,
        dbscan_eps=0.1,
        dbscan_min_samples=5,
        gt_path=gt_path,
    )
