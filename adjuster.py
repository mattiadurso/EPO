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


import glob
from PIL import Image
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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
    sample_distance_field,
    compute_chunk_loss_logic,
)
from helpers.reprojection import (
    filter_viewgraph_by_reprojection,
    grid_sample_nan,
)
from helpers.reprojection_compiled import (
    unproject_2D_to_world,
    project_world_to_2D,
    project_and_sample_logic,
)
from helpers.reconstruction import build_reconstruction
from helpers.frustum import build_view_graph_from_frustums
from modules import (
    CameraModule,
    PoseModule,
    ParameterModule,
    DepthModule,
    Edges3DModule,
)

import sys

sys.path.append("/home/mattia/Desktop/Repos/posebench/benchmarks_3D")
from benchmark_pose import eval_colmap_model


class Adjuster(nn.Module):
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
        optim (str, optional): Optimizer to use ("adamw" or "lm"). Default is "adamw".
        scheduler_name (str, optional): Learning rate scheduler name. Default is None.
        scheduler_params (dict, optional): Parameters for the learning rate scheduler.
        grad_q (bool, optional): Whether to optimize rotation quaternions. Default is True.
        grad_t (bool, optional): Whether to optimize translation vectors. Default is True.
        grad_k (bool, optional): Whether to optimize camera intrinsics. Default is True.
        grad_z (bool, optional): Whether to optimize depth scale. Default is False.
        viz (bool, optional): Whether to visualize intermediate results. Default is False.
        gt_path (str, optional): Path to ground truth data for evaluation. Default is None.
    """

    def __init__(
        self,
        reconstruction_path,
        images_path,
        depths_path,
        viewgraph_path=None,  # for testing with GT viewgraph
        unreliable_area_masks_path=None,
        lr=5e-3,
        single_camera_per_folder=True,
        load_with_pad=False,
        detector="canny",
        device="cuda",
        max_workers=-1,
        detector_params={},
        seed=0,
        max_edges_points=16_384,  # reasonabe number
        max_viewgraph_pairs=8_192,  # limit on 4090
        scheduler_name="ReduceLROnPlateau",
        scheduler_params={"factor": 0.75, "patience": 3, "min_lr": 1e-5},
        use_amp=True,
        amp_dtype=torch.bfloat16,  # torch.float16
        matcher_type="frustums",  # or "sequential"
        sequential_matcher_window=5,  # only for sequential matcher
        scene_type="outdoor",  # or "indoor", "object_centric" (not used yet)
        grad_q=True,
        grad_t=True,
        grad_k=True,
        grad_z=True,
        q_lr_scale=0.1,
        t_lr_scale=1.0,
        k_lr_scale=0.5,
        z_lr_scale=0.1,
        viz=False,  # it true del non used stuff during computation
    ):
        super().__init__()

        assert detector in [
            "canny",
            "sam2",
            "bdcn",
            "teed",
        ], f"Detector {detector} not supported."

        self.max_workers = os.cpu_count() if max_workers < 0 else max_workers
        self.images_size = 518
        self.device = device
        self.lr = lr
        self.load_with_pad = load_with_pad
        self.images_path = images_path
        self.depths_path = depths_path
        self.reconstruction_path = reconstruction_path
        self.single_camera_per_folder = single_camera_per_folder
        self.sequential_matcher_window = sequential_matcher_window
        self.convergence = False
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.dtype = torch.float32  # hardcode to float32 for stability
        self.auc_th = [1, 3, 5]
        self.auc_saving_freq = 3
        self.viewgraph_path = viewgraph_path
        self.matcher_type = matcher_type
        self.scheduler_params = scheduler_params
        self.scene_type = scene_type
        self.max_edges = max_edges_points
        self.max_viewgraph_pairs = max_viewgraph_pairs
        self.unreliable_area_masks_path = unreliable_area_masks_path
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params

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
        # what to train
        self.grad_q = grad_q
        self.grad_t = grad_t
        self.grad_k = grad_k
        self.grad_z = grad_z
        self.q_lr_scale = q_lr_scale
        self.t_lr_scale = t_lr_scale
        self.k_lr_scale = k_lr_scale
        self.z_lr_scale = z_lr_scale

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
        self.timings["load_images"] = time.time() - s_time

        ## Load poses and intrinsics
        s_time = time.time()
        # creating poses such self.poses[image_name] = PoseModel(...)
        # creating intrinsics such self.intrinsics[cam_id] = CameraModel(...)
        # using image name and camera id/foler str
        self._read_cameras_from_reconstruction()  # into self.images and self.intrinsics
        self.timings["load_poses_and_intrinsics"] = time.time() - s_time

        ## Load depth maps
        s_time = time.time()
        self._load_and_preprocess_depths()
        self.timings["load_depth_maps"] = time.time() - s_time

        ## Extract edges
        s_time = time.time()
        self._extract_edges()  # into self.images
        self.timings["extract_edges"] = time.time() - s_time

        ## Viewgraph from frustums
        s_time = time.time()
        # compute viewgraph
        self._compute_viewgraph(type=self.matcher_type)
        self.timings["compute_viewgraph"] = time.time() - s_time

        ## Compute Distance Fields
        s_time = time.time()
        self._compute_distance_fields()  # into self.images
        self.timings["compute_distance_fields"] = time.time() - s_time

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
        dt_fields = torch.stack(dt_fields, dim=0).to(self.device, dtype=self.dtype)
        images_shapes = torch.stack(images_shapes, dim=0).to(
            self.device, dtype=self.dtype
        )
        sampled_depth = torch.stack(sampled_depth, dim=0).to(
            self.device, dtype=self.dtype
        )

        # storing
        self.edges_padded = ParameterModule(
            self.image_id_map, edges_padded, self.device
        )
        self.pad_masks = ParameterModule(self.image_id_map, pad_masks, self.device)
        self.dt_fields = ParameterModule(self.image_id_map, dt_fields, self.device)
        self.images_hw = ParameterModule(self.image_id_map, images_shapes, self.device)

        self.sampled_depth = DepthModule(
            self.image_id_map,
            sampled_depth,
            self.device,
            grad=self.grad_z,
        )

        # Prepare viewgraph with indices for faster access during optimization
        viewgraph_ids = [
            (
                self.image_id_map[i],
                self.image_id_map[j],
                self.intrinsics.map_camera_ids_to_indices(self.images[i]["cam_id"]),
                self.intrinsics.map_camera_ids_to_indices(self.images[j]["cam_id"]),
            )
            for i, j in self.viewgraph
        ]

        # Also prepare image to cam id mapping tensor
        images_cams_ids = [
            (
                self.image_id_map[image_name],
                self.intrinsics.map_camera_ids_to_indices(
                    self.images[image_name]["cam_id"]
                ),
            )
            for image_name in sorted(self.images.keys())
        ]

        # (img1_id, img2_id, cam1_id, cam2_id)
        self.viewgraph_ids = torch.tensor(viewgraph_ids).long().to(self.device)
        # (image_id, cam_id)
        self.images_cams_ids = torch.tensor(images_cams_ids).long().to(self.device)

        # ==========================================================================
        # Create optimizer
        params_to_optimize = self._collect_parameters_to_optimize()
        self._print_params_summary(params_to_optimize)

        # Create optimizer with collected parameters
        self._load_optimizer(params_to_optimize)
        self._load_scheduler(self.scheduler_name, self.optimizer, self.scheduler_params)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()

        self.timings["total_loading"] = time.time() - time_start
        self.timings["total_optimization"] = 0
        self.timings["step_pre_computation"] = 0
        self.timings["prepare_batched_inputs"] = 0
        self.timings["forward_pass"] = 0
        self.timings["loss_computation"] = 0
        self.timings["gradient_computation"] = 0
        self.timings["parameter_update"] = 0
        self.timings["logging"] = 0

        # # At this point I might get rid of rgb images to save memory as not needed for edge loss
        if not viz:
            for image_name in self.images.keys():
                self.images[image_name].pop("image")
                self.images[image_name].pop("depth")
                self.images[image_name].pop("edges_map")
                if "sky_mask" in self.images[image_name]:
                    self.images[image_name].pop("sky_mask")

        self.loss_list = []
        self.lr_list = {}
        self.auc_list = {"auc": {th: [] for th in self.auc_th}, "steps": []}
        self.blacklist = set()  # Add this line

        self.seed = seed
        self.fix_seed()

        gc.collect()
        torch.cuda.empty_cache()

    def forward(
        self,
        max_steps=100,
        batch_size=128,
        residuals_chunk_size=2048,
        early_stopping=True,
        verbose=True,
        drop_last=True,
        debug=False,
        gt_path=None,
        min_lr=1e-5,
    ):
        """
        Main optimization loop.
        Args:
            max_steps (int): Maximum number of optimization steps.
            batch_size (int): Maximum number of image pairs per batch.
                            In "quick" mode: samples this many pairs and backprops immediately.
                            In "full" mode: batch size for accumulation over full viewgraph.
            early_stopping (bool): Whether to stop early if learning rate reaches minimum.
            verbose (bool): Whether to print progress.
            drop_last (bool): Whether to drop the last batch if smaller than batch_size.
        """
        time_start = time.time()

        num_batches = math.ceil(len(self.viewgraph) / batch_size)
        num_batches = num_batches if drop_last else num_batches + 1
        max_steps = max_steps if max_steps > 0 else 1_000
        if verbose:
            print(
                f"Processing {len(self.viewgraph):,} pairs with batch size {batch_size:,} ({num_batches} batches per iteration).",
                f"Using {self.images[list(self.images.keys())[0]]['edges_padded'].numel()//2:,} edges per image.",  # // due to x and y
            )

            total_points = self.max_edges * len(self.viewgraph)

            print(
                f"Total points to process per iteration: {total_points:,}.\n"
                + f"Initial learning rate: {self.lr:.2e}.\n"
                + f"Target learning rate:  {self.scheduler_params.get('min_lr',1e-4):.2e}.",
                end="\n\n",
            )

            bar = tqdm(range(max_steps), desc="Adjusting poses and intrinsics")
        else:
            bar = range(max_steps)

        for step in bar:
            t_pre = time.time()

            # Initialize optimizer gradients
            self.optimizer.zero_grad()

            # Update geometric modules
            self.poses.update_all_matrices()
            self.intrinsics.update_all_matrices()

            # Unproject point to world coordinates
            self._unproject_edges_to_3D()
            self.timings["step_pre_computation"] += time.time() - t_pre

            # Compute residuals
            residuals, sampled_viewgraphs = self.compute_forward_step(
                self.viewgraph_ids,
                batch_size=batch_size,
                drop_last=drop_last,
                residuals_chunk_size=residuals_chunk_size,
            )

            # Compute loss
            loss = self._compute_batched_loss(
                residuals, sampled_viewgraphs, debug=debug
            )

            # Backpropagate and update using GradScaler if available
            self._scaler_and_scheduler_steps(loss)

            # save lr for stopping criterion
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.lr
            )
            # ============================================================
            # Logging
            # ============================================================
            logging_time_start = time.time()
            self.loss_list.append(loss.detach().item())

            # self.lr_list.append(current_lr)
            # store lr for each group
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i not in self.lr_list:
                    self.lr_list[i] = []
                self.lr_list[i].append(param_group["lr"])

            # DEBUG: Evaluate AUC if GT available
            if gt_path is not None and step % self.auc_saving_freq == 0 and step > 0:

                opt = "optimized_reconstruction_GD/_current_test"
                self.to_colmap(opt, save_points=False, verbose=False)
                AUC_score_max, num_images, df_optim = eval_colmap_model(
                    opt, gt_path, return_df=False, thrs=self.auc_th
                )
                # store AUC
                for i, th in enumerate(self.auc_th):
                    self.auc_list["auc"][th].append(AUC_score_max[i].item())
                self.auc_list["steps"].append(step)

            if verbose:
                bar.set_postfix(
                    loss=f"{self.loss_list[-1]:.4f}",
                    auc5=(
                        f"{self.auc_list['auc'][self.auc_th[-1]][-1]:.4f}"
                        if len(self.auc_list["auc"][self.auc_th[-1]]) > 0
                        else 0
                    ),
                    lr=f"{current_lr:.2e}",
                )

                self.timings["logging"] += time.time() - logging_time_start

            # Stopping criterion: stop when lr stops decreasing | change for cosine annealing
            if early_stopping and current_lr <= self.scheduler_params.get(
                "min_lr", min_lr
            ):
                print(
                    f"Learning rate reached minimum threshold {current_lr:.2e}"
                    + f" <= {self.scheduler_params.get('min_lr', 1e-4):.2e}. "
                    + "Stopping optimization."
                )
                break

        self.timings["total_optimization"] += time.time() - time_start
        self.print_summary() if verbose else None

    ### Forward and backward helpers ###

    def _unproject_edges_to_3D(self, batch_size=None):
        """Unproject 2D edges to 3D points for all images as a batch."""
        image_names_id = self.images_cams_ids[:, 0]
        cam_ids = self.images_cams_ids[:, 1]

        # indexing data
        K_batch = self.intrinsics.get_intrinsic_matrix_inverse(cam_ids)  # (B, 3, 3)
        P_batch = self.poses.get_projection_matrix_inverse(image_names_id)  # (B, 4, 4)
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
                xy0=xy0, K0=K0, depth0=depth0, P0=P0, skip_inversion=True
            )  # (bs, N, 3)
            points_3D_list.append(pts3d)

        points_3D = torch.cat(points_3D_list, dim=0)

        # Store points_3D in Edges3DModule
        self.edges_3D = Edges3DModule(self.image_id_map, points_3D, self.device)

    def _create_batched_inputs(self, sampled_viewgraph):
        """Prepare batched inputs for the batched optimization step given a list of pairs from the viewgraph."""
        # mantain str index for visualization/debugging
        if isinstance(sampled_viewgraph[0][0], str):
            cam_ids = []
            images_names_ij = []
            images_names_ji = []
            for (
                i,
                j,
            ) in sampled_viewgraph:  # with i,j being left (0) and right (1) images
                # I already have unprojected points to 3D, so I only have to check that
                # those points reproject within the image boundaries.

                # these are are the ids for points 3D and their corresponding pad
                images_names_ij.append(i)
                images_names_ij.append(j)

                # these are the ids for right images where to reproject
                cam_ids.append(self.images[j]["cam_id"])
                cam_ids.append(self.images[i]["cam_id"])
                images_names_ji.append(j)
                images_names_ji.append(i)

        else:
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
        # reduce viewgraph if too large, use torch
        if len(sampled_viewgraph) > self.max_viewgraph_pairs:
            indices = torch.randperm(len(sampled_viewgraph))[: self.max_viewgraph_pairs]
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
            batch, pad_masks, dt_fields = self._create_batched_inputs(sampled_viewgraph)
            self.timings["prepare_batched_inputs"] += time.time() - s_time

            # actual inference
            s_time = time.time()
            # actual inference
            with torch.amp.autocast(
                device_type=self.device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
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

    def _compute_batched_loss(
        self, residuals, sampled_viewgraphs=None, debug=False, delta=1.0
    ):
        """Vectorized batched loss computation."""
        s_time = time.time()

        if residuals.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        residuals_pairs = residuals.view(-1, 2)

        # clamp residuals. 15 is quite high already
        residuals_pairs = torch.clamp(residuals_pairs, min=0, max=10.0)

        # Use Huber loss for robust cost
        pair_losses = F.huber_loss(
            residuals_pairs,
            torch.zeros_like(residuals_pairs),
            reduction="none",
            delta=delta,
        )
        pair_losses = residuals_pairs.sum(dim=1)  # (num_pairs,)

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
            cam_id=cam_id_to_tensor_id,
            k_models=k_models,
            k_params=torch.stack(k_params),
            k_grad=self.grad_k,
            device=self.device,
            dtype=self.dtype,
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
            R=torch.stack(R_tensor),
            t=torch.stack(t_tensor),
            grad_q=self.grad_q,
            grad_t=self.grad_t,
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
                    depth = F.pad(depth, pad, mode="constant", value=depth.max())
                    self.images[image_name]["depth"] = depth.to(
                        self.device, dtype=self.dtype
                    )

    # In your Adjuster class, before calling filter_viewgraph_by_reprojection:

    @torch.no_grad()
    def _compute_viewgraph(self, type="frustums"):
        """Compute viewgraph and filter by reprojection error and returns the sorted viewgraph."""
        if type == "frustums":
            # Estimate view graph from frustums
            viewgraph = build_view_graph_from_frustums(
                self.recon,
                max_view_angle_deg=30.0,
                distance_factor=2,
                verbose=False,
                images_with_depth=self.images,
                dtype=self.dtype,
            )

        elif type == "sequential":
            # Build sequential viewgraph based on sorted image names with a window size of 10
            image_names = sorted(list(self.images.keys()))
            viewgraph = []
            for i in range(len(image_names) - self.sequential_matcher_window):
                for j in range(1, self.sequential_matcher_window + 1):
                    viewgraph.append((image_names[i], image_names[i + j]))

        elif type == "exhaustive":
            # Build exhaustive viewgraph (all pairs)
            image_names = sorted(list(self.images.keys()))
            viewgraph = []
            for i in range(len(image_names)):
                for j in range(i + 1, len(image_names)):
                    viewgraph.append((image_names[i], image_names[j]))

        else:
            raise ValueError(f"Viewgraph type {type} not supported.")

        # Filter viewgraph by reprojection
        # min_points shouldbe set such images have enough overlap, maybe at least 25%
        # 518*345*0.25 = 44_677
        filtered_viewgraph = filter_viewgraph_by_reprojection(
            self,
            viewgraph,
            self.images,
            min_points=100,  # 10_000
            sampling_factor=5,  # 1
            reprojection_error=5.0,  # 5
            border=10,
            use_amp=self.use_amp,
        )

        self.viewgraph = filtered_viewgraph
        self.viewgraph.sort(key=lambda x: (x[0], x[1]))

    def filter_images_by_registration_time(self, threshold=1 / 3):
        """
        Blacklist images with poor registration times.
        These images will be skipped during optimization and reconstruction.

        Args:
            threshold (float): Threshold for filtering (default 0.3)

        Returns:
            list: Names of blacklisted images
        """
        blacklisted = [
            img_name
            for img_name, reg_time in self.times_image_was_registered.items()
            if reg_time < threshold
        ]

        if blacklisted:
            self.blacklist = set(blacklisted)
            # Filter viewgraph
            original_pairs = len(self.viewgraph)
            new_viewgraph = [
                (i, j)
                for i, j in self.viewgraph
                if i not in self.blacklist and j not in self.blacklist
            ]
            self.viewgraph = new_viewgraph
            removed_pairs = original_pairs - len(self.viewgraph)
            print(
                f"Blacklisted {len(blacklisted)} images with registration time < {threshold:.3f}"
            )
            print(f"Removed {removed_pairs:,} viewgraph pairs")
        else:
            print(f"No images found with registration time < {threshold:.3f}")

    ### Edges
    def _extract_edges(self, min_percentile=10, max_percentile=75):
        for image_name in tqdm(self.images.keys(), desc=f"Extracting edges"):
            img_tensor = self.images[image_name]["image"].unsqueeze(0)
            edges_map = self.edge_extractor(img_tensor)

            # --- NEW: Filter edges at depth discontinuities for object-centric scenes ---
            if (
                self.scene_type == "object_centric"
                and "depth" in self.images[image_name]
            ):
                depth = self.images[image_name]["depth"].squeeze()  # (H, W)

                valid_depth_mask = (depth > 0) & (~torch.isnan(depth))
                depth_jump_mask = torch.zeros_like(depth, dtype=torch.bool)

                # 2. Filter point too close or too far depth wise
                background_mask = torch.zeros_like(depth, dtype=torch.bool)
                if valid_depth_mask.any():
                    # Compute 75th percentile of valid depth values
                    valid_depths = depth[valid_depth_mask]
                    depth_p75 = torch.quantile(valid_depths, max_percentile / 100)
                    background_mask = (depth > depth_p75) & valid_depth_mask

                    # exclude also very close depths (e.g., depth < 5th percentile)
                    depth_p5 = torch.quantile(valid_depths, min_percentile / 100)
                    background_mask = background_mask | (depth < depth_p5)

                # Combine masks: remove jumps AND background
                mask_to_remove = depth_jump_mask | background_mask

                # Apply mask
                edges_map = edges_map.squeeze()  # (H, W)
                # Use multiplication for masking float tensor
                edges_map = edges_map * (~mask_to_remove).float()
                edges_map = edges_map.unsqueeze(0)  # Restore (1, H, W)
            # ----------------------------------------------------------------------------

            edges = edges_map.squeeze().nonzero().flip(dims=(1, 0))  # (N, 2)
            self.images[image_name].update(
                {
                    "edges_map": edges_map.squeeze().to(self.device, dtype=self.dtype),
                    "edges": edges.to(self.device, dtype=self.dtype),
                }
            )

        # if binary mask with unreliable ares are provided, load and apply to edge_maps
        if self.unreliable_area_masks_path is not None:
            # load images
            for image_name in tqdm(
                self.images.keys(), desc="Applying unreliable area masks"
            ):
                mask_path = os.path.join(
                    self.unreliable_area_masks_path,
                    image_name.split(".")[0] + "_mask.png",
                )
                if os.path.exists(mask_path):
                    mask_pil = Image.open(mask_path).convert("L")
                    mask_tensor = (
                        torch.from_numpy(np.array(mask_pil)).to(self.device).bool()
                    )  # (1, H, W)
                    mask_tensor = TF.resize(
                        mask_tensor.unsqueeze(0),
                        self.images[image_name]["edges_map"].shape,
                        interpolation=TF.InterpolationMode.NEAREST,
                    ).squeeze()

                    # Apply mask to edges_map
                    edges_map = self.images[image_name]["edges_map"]
                    edges_map = edges_map * ~mask_tensor
                    self.images[image_name]["edges_map"] = edges_map

                    # Re-extract edges
                    edges = edges_map.squeeze().nonzero().flip(dims=(1, 0))  # (N, 2)
                    self.images[image_name]["edges"] = edges.to(
                        self.device, dtype=self.dtype
                    )
                    # add mask for logging
                    self.images[image_name]["unreliable_area_mask"] = mask_tensor

                else:
                    print(f"Warning: Unreliable area mask not found for {mask_path}")

        # pad to have same number of edges per image
        self._pad_edges()

        # add sampled depth at edges_padded locations
        for image_name in self.images.keys():
            edges_padded = self.images[image_name]["edges_padded"]  # (N, 2)
            depth = self.images[image_name]["depth"]  # (H, W)
            sampled_depth, _ = grid_sample_nan(edges_padded[None], depth[None])
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
            torch.quantile(torch.tensor(num_edges, dtype=self.dtype), 0.9).long().item()
        )

        # this to save some computation/memory
        # likely only few images have very large number of edges
        max_edges_to_retain = min(q90, max_edges)

        for image_name in self.images.keys():
            edges = self.images[image_name]["edges"]
            n_edges = edges.shape[0]

            if n_edges > max_edges_to_retain:
                # randomly sample max_edges
                indices = torch.randperm(n_edges, device=edges.device)[
                    :max_edges_to_retain
                ]
                edges = edges[indices]
                n_edges = max_edges_to_retain

            if n_edges < max_edges_to_retain:
                pad_size = max_edges_to_retain - n_edges
                pad = torch.zeros((pad_size, 2), device=edges.device)
                edges = torch.cat([edges, pad], dim=0)

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
        print(
            f"Edges stats:\n",
            f"max: {max_edges:,} |",
            f"min: {min_edges:,} | avg: {int(avg_edges):,} |",
            f"std: {std_edges:.2f} |",
            f"quantiles (0.5, 0.9): {median_edges:,}, {q90:,}",
        )

    @torch.no_grad()
    def _compute_distance_fields(self):
        dt_fields_shapes = []
        for image_name in tqdm(self.images.keys(), desc="Computing distance fields"):
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

    ## Optimizer and scheduler
    def _collect_parameters_to_optimize(self):
        params_to_optimize = {}

        # Collect parameters to optimize
        if self.grad_k:
            params_to_optimize["k"] = self.intrinsics.parameters()

        if self.grad_q:
            params_to_optimize["q"] = self.poses.parameters(q=True, t=False)

        if self.grad_t:
            params_to_optimize["t"] = self.poses.parameters(q=False, t=True)

        if self.grad_z:
            params_to_optimize["z"] = self.sampled_depth.parameters()

        return params_to_optimize

    def _print_params_summary(self, params_to_optimize):
        total_params = 0
        print("\nTotal parameters to optimize:")
        for key in ["k", "t", "q", "z", "mlp"]:
            space = 14 if key == "mlp" else 16
            if key not in params_to_optimize:
                print(f"  {key}: {0:>{space},}")
                continue
            set_params = sum(p.numel() for p in params_to_optimize[key])
            print(f"  {key}: {set_params:>{space},}")
            total_params += set_params
        print("-" * 23)
        print(f"  {'Total':}: {total_params:>12,}\n")

    def _load_optimizer(self, params):
        param_groups = []
        if "t" in params:
            param_groups.append(
                {"params": params["t"], "lr": self.lr * self.t_lr_scale}
            )
        if "k" in params:
            param_groups.append(
                {"params": params["k"], "lr": self.lr * self.k_lr_scale}
            )
        if "z" in params:
            param_groups.append(
                {"params": params["z"], "lr": self.lr * self.z_lr_scale}
            )
        if "q" in params:
            param_groups.append(
                {"params": params["q"], "lr": self.lr * self.q_lr_scale}
            )

        self.optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

    def _load_scheduler(self, name, optimizer, params=None):
        """
        Dynamically loads a PyTorch scheduler by name.

        Args:
            name (str): Name of the scheduler class (e.g., 'StepLR', 'CosineAnnealingLR').
            optimizer (torch.optim.Optimizer): The optimizer instance to attach the scheduler to.
            params (dict, optional): Keyword arguments for the scheduler (e.g., step_size, gamma).

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The initialized scheduler.
        """
        if not name:
            self.scheduler = None
            return

        params = {} if params is None else params

        # Retrieve all available schedulers in torch.optim.lr_scheduler
        schedulers = {
            cls_name.lower(): cls
            for cls_name, cls in torch.optim.lr_scheduler.__dict__.items()
            if isinstance(cls, type)
        }

        key = name.lower()
        if key not in schedulers:
            raise ValueError(
                f"Unknown scheduler '{name}'. Available: {list(schedulers.keys())}"
            )

        SchedulerClass = schedulers[key]
        self.scheduler = SchedulerClass(optimizer, **params)

        print(f"Using scheduler: {name} with params: {params}")

    def _scheduler_step(self, loss):
        if self.scheduler is not None:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

    def _scaler_and_scheduler_steps(self, loss, clip_gradients=False):
        # scaler step
        if hasattr(self, "scaler") and self.scaler is not None:
            # gradients computation
            s_time = time.time()
            self.scaler.scale(loss).backward()
            self.timings["gradient_computation"] += time.time() - s_time

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # parameter update
            s_time = time.time()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.timings["parameter_update"] += time.time() - s_time
        else:
            # Gradients computation
            s_time = time.time()
            loss.backward()
            self.timings["gradient_computation"] += time.time() - s_time

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Parameter update
            s_time = time.time()
            self.optimizer.step()
            self.timings["parameter_update"] += time.time() - s_time

        # scheduler step
        if self.scheduler is not None:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(loss.detach())
            else:
                self.scheduler.step()

    ### Reconstruction
    def to_colmap(
        self,
        output_path="optimized_reconstruction_GD",
        save_points=False,
        verbose=False,
        max_points_per_image=100_000,
        final_dbscan_filtering=False,
        dbscan_eps=0.05,
        dbscan_min_samples=5,
    ):
        recon = build_reconstruction(
            self,
            output_path=output_path,
            save_points=save_points,
            verbose=verbose,
            max_points_per_image=max_points_per_image,
            final_dbscan_filtering=final_dbscan_filtering,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

        # save loading time and optimization time in timings.txt in same folder as output_path
        timings_path = os.path.join(output_path, "timings.txt")
        with open(timings_path, "w") as f:
            for key, value in self.timings.items():
                f.write(f"{key}: {value:.4f} s\n")
        return recon

    ### Misc
    def print_summary(self, w=None):
        # Column widths
        key_width = 30
        val_width = 10
        perc_width = 8
        avg_width = 12

        # Total line width
        if w is None:
            w = key_width + val_width + perc_width + avg_width + 6

        print("\n" + "=" * w)
        print(f"{'Summary':^{w}}")
        print("-" * w)

        # Compute total
        self.timings["total"] = self.timings.get("total_loading", 0) + self.timings.get(
            "total_optimization", 0
        )

        # Header row
        print(
            f"{'Stage':<{key_width}}"
            f"{'Time (s)':>{val_width}}"
            f"{'%':>{perc_width+1}}"
            f"{'Per Iter':>{avg_width}}"
        )
        print("-" * w)

        num_iters = len(getattr(self, "loss_list", []))

        ordered_keys = [
            "total_loading",
            "step_pre_computation",
            "prepare_batched_inputs",
            "forward_pass",
            "loss_computation",
            "gradient_computation",
            "parameter_update",
            "logging",
            "total_optimization",
        ]

        per_iter_keys = {
            "step_pre_computation",
            "prepare_batched_inputs",
            "forward_pass",
            "loss_computation",
            "gradient_computation",
            "parameter_update",
            "logging",
        }

        for key in ordered_keys:
            if key not in self.timings:
                continue

            value = self.timings[key]

            if value == 0 and key not in per_iter_keys:
                continue

            if key in per_iter_keys and num_iters > 0:
                # Show total time, percentage, AND per-iteration average
                perc = (
                    (value / self.timings["total"]) * 100
                    if self.timings["total"] > 0
                    else 0
                )
                value_avg = value / num_iters
                row_str = (
                    f"{key:<{key_width}}"
                    f"{value:>{val_width}.2f}"
                    f"{perc:>{perc_width}.1f}%"
                    f"{value_avg:>{avg_width}.4f}"
                )
            else:
                # Show total time and percentage
                perc = (
                    (value / self.timings["total"]) * 100
                    if self.timings["total"] > 0
                    else 0
                )
                row_str = (
                    f"{key:<{key_width}}"
                    f"{value:>{val_width}.2f}"
                    f"{perc:>{perc_width}.1f}%"
                    f"{'':>{avg_width}}"
                )

            print(row_str)

        print("-" * w)
        print(
            f"{'Total':<{key_width}}"
            f"{self.timings['total']:>{val_width}.2f}"
            f"{'':>{perc_width + avg_width + 3}}"
        )

        # Loss summary
        if len(self.loss_list) > 0:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss

            print("-" * w)
            print(f"{'Initial loss:':<{key_width}}{initial_loss:>{val_width}.6f}")
            print(f"{'Final loss:':<{key_width}}{final_loss:>{val_width}.6f}")
            # loss and percentage with sign under % column on same row
            perc_improvement = (
                (delta / initial_loss) * 100 if initial_loss != 0 else 0.0
            )
            sign = "-" if perc_improvement > 0 else "+"
            perc_improvement = sign + f"{abs(perc_improvement):.1f}"
            print(
                f"{'Loss reduction:':<{key_width}}{delta:>{val_width}.6f}"
                f"{perc_improvement:>{perc_width}}%"
            )
            steps = len(self.loss_list)
            conv = " (converged)" if getattr(self, "convergence", False) else ""
            print(f"{f'Total steps{conv}:':<{key_width}}{steps:>{val_width}d}")

        print("=" * w)

    def fix_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __repr__(self):
        repr_str = f"Adjuster(\n"
        repr_str += f"  Reconstruction path: {self.reconstruction_path}\n"
        repr_str += f"  Images path: {self.images_path}\n"
        repr_str += f"  Depths path: {self.depths_path}\n"
        repr_str += f"  Number of images: {len(self.images)}\n"
        if hasattr(self, "viewgraph"):
            repr_str += f"  Number of viewgraph edges: {len(self.viewgraph):,}\n"

        total_params = 0
        params_to_optimize = self._collect_parameters_to_optimize()
        for key in ["k", "t", "q", "z"]:
            if key in params_to_optimize:
                set_params = sum(p.numel() for p in params_to_optimize[key])
                total_params += set_params

        repr_str += f"  Total parameters to optimize: {total_params:,}\n"
        converged_str = " (converged)" if getattr(self, "convergence", False) else ""
        repr_str += (
            f"  Number of optimization steps: {len(self.loss_list)}{converged_str}\n"
        )

        if len(self.loss_list) >= 2:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss
            perc_improvement = (
                (delta / initial_loss) * 100 if initial_loss != 0 else 0.0
            )
            repr_str += f"  Loss improvement: {delta:.3f} ({perc_improvement:.2f}%)\n"

        repr_str += f")"
        return repr_str

    @torch.no_grad()
    def visualize_residuals(
        self,
        output_dir="residual_maps",
        percentile=99,
        max_images=100,
        custom_viewgraph=None,
    ):
        """
        Visualize reprojection residuals for all image pairs in the viewgraph.
        Creates error maps showing where edges align well or poorly between image pairs.

        Args:
            output_dir (str): Directory to save residual visualization maps
            percentile (float): Percentile for colormap scaling (default 95 to avoid outliers)
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        os.makedirs(output_dir, exist_ok=True)

        # Select viewgraph pairs
        if custom_viewgraph:
            viewgraph = custom_viewgraph
        else:
            viewgraph = (
                self.viewgraph
                if max_images < 0
                else random.choices(self.viewgraph, k=max_images)
            )
        num_pairs = len(viewgraph)
        print(f"Visualizing residuals for {num_pairs:,} image pairs...")

        for pair_idx, (img_i, img_j) in enumerate(
            tqdm(viewgraph, desc="Computing residuals")
        ):
            sampled_vg = [(img_i, img_j)]
            batch, pad_masks, dt_fields = self._create_batched_inputs(sampled_vg)
            # Project edges and compute residuals
            edges_reprojected, _ = project_world_to_2D(**batch)
            residuals = sample_distance_field(dt_fields, edges_reprojected).squeeze(1)

            # Forward: i->j
            res_ij = residuals[0]  # (N,)
            edges_ij = edges_reprojected[0]  # (N, 2)
            pad_mask_j = self.images[img_j]["pad_mask"]
            img_j_tensor = (
                self.images[img_j]["image"]
                if "image" in self.images[img_j]
                else torch.zeros(3, *self.images[img_j]["hw"])
            )
            edges_map_j = self.images[img_j]["edges_map"].cpu().numpy()
            hw_j = self.images[img_j]["hw"]

            # Backward: j->i
            res_ji = residuals[1]
            edges_ji = edges_reprojected[1]
            pad_mask_i = self.images[img_i]["pad_mask"]
            img_i_tensor = (
                self.images[img_i]["image"]
                if "image" in self.images[img_i]
                else torch.zeros(3, *self.images[img_i]["hw"])
            )
            edges_map_i = self.images[img_i]["edges_map"].cpu().numpy()
            hw_i = self.images[img_i]["hw"]

            # Save visualization
            self._save_residual_visualization_custom(
                img_i_tensor,
                edges_map_i,
                img_j_tensor,
                edges_map_j,
                edges_ij,
                res_ij,
                pad_mask_j,
                hw_j,
                edges_ji,
                res_ji,
                pad_mask_i,
                hw_i,
                img_i,
                img_j,
                output_dir,
                pair_idx,
                percentile,
            )

    @torch.no_grad()
    def _save_residual_visualization_custom(
        self,
        img_i_tensor,
        edges_map_i,
        img_j_tensor,
        edges_map_j,
        edges_ij,
        res_ij,
        pad_mask_j,
        hw_j,
        edges_ji,
        res_ji,
        pad_mask_i,
        hw_i,
        img_i,
        img_j,
        output_dir,
        pair_idx,
        percentile,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        # Prepare filenames
        safe_img_i = img_i.replace("/", "_").replace("\\", "_")
        safe_img_j = img_j.replace("/", "_").replace("\\", "_")
        filename = f"{pair_idx:04d}_{safe_img_i}_to_{safe_img_j}.png"
        filepath = os.path.join(output_dir, filename)

        # Normalize images
        img_i_np = img_i_tensor.cpu().numpy().transpose(1, 2, 0)
        img_j_np = img_j_tensor.cpu().numpy().transpose(1, 2, 0)
        img_i_np = np.clip(
            img_i_np / (img_i_np.max() if img_i_np.max() > 1.0 else 1.0), 0, 1
        )
        img_j_np = np.clip(
            img_j_np / (img_j_np.max() if img_j_np.max() > 1.0 else 1.0), 0, 1
        )

        # Edge maps: white edges on black
        edges_img_i = np.zeros((*hw_i, 3), dtype=np.float32)
        edges_img_i[edges_map_i > 0] = 1.0
        edges_img_j = np.zeros((*hw_j, 3), dtype=np.float32)
        edges_img_j[edges_map_j > 0] = 1.0

        # Row 1, Col 3: image i edges (white) + projected edges from j (colored)
        combined_i = np.zeros((*hw_i, 3), dtype=np.float32)
        combined_i[edges_map_i > 0] = 1.0  # white edges from i
        valid_mask = pad_mask_i > 0.5
        valid_edges = edges_ji[valid_mask].long().cpu().numpy()
        valid_residuals = res_ji[valid_mask].cpu().numpy()
        if valid_edges.shape[0] > 0:
            vmax = np.percentile(valid_residuals, percentile)
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = cm.get_cmap("RdYlGn_r")
            for idx, (x, y) in enumerate(valid_edges):
                color = cmap(norm(valid_residuals[idx]))[:3]
                x = np.clip(x, 0, hw_i[1] - 1)
                y = np.clip(y, 0, hw_i[0] - 1)
                combined_i[y, x] = color

        # Row 2, Col 3: image j edges (white) + projected edges from i (colored)
        combined_j = np.zeros((*hw_j, 3), dtype=np.float32)
        combined_j[edges_map_j > 0] = 1.0  # white edges from j
        valid_mask = pad_mask_j > 0.5
        valid_edges = edges_ij[valid_mask].long().cpu().numpy()
        valid_residuals = res_ij[valid_mask].cpu().numpy()
        if valid_edges.shape[0] > 0:
            vmax = np.percentile(valid_residuals, percentile)
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = cm.get_cmap("RdYlGn_r")
            for idx, (x, y) in enumerate(valid_edges):
                color = cmap(norm(valid_residuals[idx]))[:3]
                x = np.clip(x, 0, hw_j[1] - 1)
                y = np.clip(y, 0, hw_j[0] - 1)
                combined_j[y, x] = color

        # Compute mean residual for display (as in loss), clmap and apply huber loss
        # mean_residual = 0.5 * (res_ij.mean().item() + res_ji.mean().item())
        max_residual_ij = res_ij.max().item()
        max_residual_ji = res_ji.max().item()

        res_ij = res_ij.clamp(max=10.0)
        res_ji = res_ji.clamp(max=10.0)
        delta = 1.0
        huber_ij = (
            0.5 * res_ij**2 * (res_ij <= delta).float()
            + (delta * (res_ij - 0.5 * delta)) * (res_ij > delta).float()
        )
        huber_ji = (
            0.5 * res_ji**2 * (res_ji <= delta).float()
            + (delta * (res_ji - 0.5 * delta)) * (res_ji > delta).float()
        )
        mean_residual = 0.5 * (huber_ij.mean().item() + huber_ji.mean().item())

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # Reduced size
        plt.subplots_adjust(wspace=0.08, hspace=0.08)  # Less space between columns/rows

        # Add residual value in top-left corner (black text)
        fig.text(
            0.12,
            0.9,
            f"Residual: {mean_residual:.3f}, Max residuals: {max_residual_ij:.3f} and {max_residual_ji:.3f}",
            ha="left",
            va="top",
            color="black",
            fontsize=10,
            weight="bold",
        )

        # Row 1: image i, edges i, edges i + projected edges from j
        axes[0, 0].imshow(img_i_np)
        axes[0, 0].set_title(f"Image: {img_i}")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(edges_img_i)
        axes[0, 1].set_title(f"Edges: {img_i}")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(combined_i)
        axes[0, 2].set_title(f"Edges {img_i} + proj. edges from {img_j}")
        axes[0, 2].axis("off")

        # Row 2: image j, edges j, edges j + projected edges from i
        axes[1, 0].imshow(img_j_np)
        axes[1, 0].set_title(f"Image: {img_j}")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(edges_img_j)
        axes[1, 1].set_title(f"Edges: {img_j}")
        axes[1, 1].axis("off")
        axes[1, 2].imshow(combined_j)
        axes[1, 2].set_title(f"Edges {img_j} + proj. edges from {img_i}")
        axes[1, 2].axis("off")

        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    scene = "vienna_state_opera"
    which_data = "data_test" if scene == "graz_main_square" else "data"

    reconstruction_path = (
        f"/home/mattia/Desktop/Repos/vggt/wrapper_output/{scene}/sparse"
    )
    images_path = f"/home/mattia/Desktop/datasets/mydataset/{which_data}/{scene}/frames"
    depths_path = (
        f"/home/mattia/Desktop/Repos/vggt/wrapper_output/{scene}/sparse/depth_maps"
    )
    gt_path = (
        f"/home/mattia/Desktop/datasets/mydataset/{which_data}/{scene}/colmap/sparse/0"
    )
    # unreliable_area_masks_path = images_path.replace(dataset_cfg["images_folder"], "depth_masks_mask2former")

    adjuster = Adjuster(
        reconstruction_path=reconstruction_path,
        images_path=images_path,
        depths_path=depths_path,
        # unreliable_area_masks_path=unreliable_area_masks_path,
        single_camera_per_folder=True,
        grad_k=True,
        grad_q=True,
        grad_t=True,
        grad_z=True,
        detector="teed",  # or "canny", "bdcn", "sam2"
        # # outdoor
        lr=5e-4,
        matcher_type="frustums",  # or "exhaustive", "sequential"
        scheduler_params={"factor": 0.75, "patience": 3, "min_lr": 1e-4},
        detector_params={
            "low_threshold": 0.20,
            "high_threshold": 0.25,
            "kernel_size": 7,
            "sigma": 2,
        },
        viz=True,
    )

    adjuster(
        batch_size=256,
        max_steps=-1,
        debug=True,  # tracks the residuals, slightly increases timing
        gt_path=gt_path,
    )
