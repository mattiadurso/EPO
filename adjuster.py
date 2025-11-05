# This script should contain the pose, intrinsics and depth optimizer.
# It should work with PyTorch and be compatible with CUDA if available.
# It should works usign edges and photometric losses.

# I might want to use pypose, kornia and/or pytorch3d for this.
import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import pycolmap
import warnings
import random

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


from tqdm import tqdm
from helpers.load import (
    find_images,
    load_and_preprocess_images,
    load_and_preprocess_depths,
)
from losses.loss import (
    compute_distance_field,
    sample_distance_field,
)
from helpers.reprojection import filter_viewgraph_by_reprojection, reproject_2D_2D
from helpers.frustum import build_view_graph_from_frustums
from extractors.canny import CannyEdgeDetector
from modules.camera import Camera
from modules.pose import Pose


class Adjuster(nn.Module):
    """
    Currently I am at an early stage of implementing this class. So i'll just describe what it should do.

    Inputs:
        - images: A batch of images (B, C, H, W)
        - depth_maps: Initial depth maps (B, 1, H, W)
        - pycolmap.Reconstruction: A pycolmap reconstruction object containing the initial poses and intrinsics
                                    (or directly the poses and intrinsics in torch)

    Outputs:
        optimized depth maps, updated pycolmap.Reconstruction
    """

    def __init__(
        self,
        reconstruction,
        images_path,
        depths_path,
        lr=1e-3,
        single_camera_per_folder=True,
        load_with_pad=True,
        detector="canny",
        device="cuda",
        max_workers=-1,
        detector_params={},
        seed=0,
        optim="adamw",  # or "LM"
        grad_q=True,
        grad_t=True,
        grad_k=True,
    ):
        super().__init__()

        assert detector in ["canny"], f"Detector {detector} not supported."
        assert optim.lower() in [
            "adamw",
            "lm",
        ], f"Optimizer {optim} not supported."
        self.use_pypose = False

        self.max_workers = os.cpu_count() if max_workers < 0 else max_workers
        self.images_size = 518  # square size to which images are resized
        self.device = device
        self.lr = lr

        # Edge extractor
        if detector == "canny":
            self.edge_extractor = CannyEdgeDetector(
                low_threshold=detector_params.get("low_threshold", 0.15),
                high_threshold=detector_params.get("high_threshold", 0.25),
                hysteresis=detector_params.get("hysteresis", True),
                kernel_size=detector_params.get("kernel_size", 5),
                sigma=detector_params.get("sigma", 3.0),
                device=device,
            )

        # what to train
        self.grad_q = grad_q
        self.grad_t = grad_t
        self.grad_k = grad_k

        # Loading
        self.timings = {}
        time_start = time.time()

        recon = pycolmap.Reconstruction(reconstruction)

        ## Load Images as dict {image_name: image_tensor}
        s_time = time.time()
        image_path_list = find_images(images_path)
        images = load_and_preprocess_images(
            image_path_list,
            images_path,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=load_with_pad,
            device=self.device,
        )
        self.timings["load_images"] = time.time() - s_time

        ## Load poses and intrinsics
        s_time = time.time()
        images, intrinsics = self.read_cameras_from_reconstruction(
            recon,
            images,
            single_camera_per_folder=single_camera_per_folder,
            load_with_pad=load_with_pad,
        )
        self.timings["load_poses_and_intrinsics"] = time.time() - s_time

        ## Load depth maps
        s_time = time.time()
        images = load_and_preprocess_depths(
            depths_path,
            images,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=load_with_pad,
            device=self.device,
        )
        self.timings["load_depth_maps"] = time.time() - s_time

        ## Extract edges
        s_time = time.time()
        for image_name in tqdm(images.keys(), desc=f"Extracting edges"):
            img_tensor = images[image_name]["image"].unsqueeze(0).to(self.device)
            edges_map = self.edge_extractor(img_tensor)
            edges = edges_map.squeeze().nonzero().flip(dims=(1, 0)).float()  # (N, 2)
            images[image_name].update(
                {"edges_map": edges_map.squeeze(), "edges": edges}
            )
        self.timings["extract_edges"] = time.time() - s_time

        ## Compute Distance Fields
        s_time = time.time()
        for image_name in tqdm(images.keys(), desc="Computing distance fields"):
            edges_map = images[image_name]["edges_map"]
            dt_field = compute_distance_field(
                edges_map,
                device=self.device,
            )
            images[image_name].update({"dt_field": dt_field})
        self.timings["compute_distance_fields"] = time.time() - s_time

        ## Viewgraph from frustums
        s_time = time.time()
        # Estimate view graph from frustums
        viewgraph = build_view_graph_from_frustums(
            recon,
            z_near_default=0.1,
            z_far_default=5.0,
            max_view_angle_deg=30.0,
            distance_factor=2,
            verbose=False,
        )
        # Filter viewgraph by reprojection | This need to be runned as batch and speeded up
        viewgraph = filter_viewgraph_by_reprojection(
            viewgraph,
            images,
            intrinsics,
            th=0.025,
            sampling_factor=10,
            reprojection_error=3.0,
        )
        self.timings["viewgraph"] = time.time() - s_time

        self.images = images
        self.viewgraph = viewgraph
        self.intrinsics = intrinsics

        # Create optimizer
        params_to_optimize = {}

        # Collect parameters to optimize
        if self.grad_k:
            k_params = []
            for camera in self.intrinsics.values():
                k_params.extend(camera.parameters())
            params_to_optimize["k"] = k_params

        if self.grad_q:
            q_params = []
            for image_name, image_data in self.images.items():
                q_params.extend(
                    image_data["P"].parameters(q=True, t=False) if self.grad_q else []
                )
            params_to_optimize["q"] = q_params

        if self.grad_t:
            t_params = []
            for image_name, image_data in self.images.items():
                t_params.extend(
                    image_data["P"].parameters(q=False, t=True) if self.grad_t else []
                )
            params_to_optimize["t"] = t_params

        total_params = 0
        print("\nTotal parameters to optimize:")
        for key in ["k", "t", "q", "z"]:
            if key not in params_to_optimize:
                print(f"  {key}: {0:>16,} parameters")
                continue
            set_params = sum(p.numel() for p in params_to_optimize[key])
            print(f"  {key}: {set_params:>16,} parameters")

        print(f"  {'Total':}: {total_params:>12,} parameters\n")

        # Create optimizer with collected parameters
        self.optimizer = self.load_optimizer(optim, params_to_optimize)
        self.scaler = torch.cuda.amp.GradScaler(enabled=not self.use_pypose)

        self.timings["total_loading"] = time.time() - time_start
        self.timings["total_optimization"] = 0

        # # At this point I might get rid of rgb images to save memory as not needed for edge loss
        # for image_name in self.images.keys():
        #     self.images[image_name].pop("image")

        self.loss_list = []
        self.seed = seed
        self.fix_seed()

        gc.collect()
        torch.cuda.empty_cache()

    def forward(
        self,
        max_steps=100,
        quick_mode=True,
        max_pairs=512,
    ):
        """
        Main optimization loop.
        Args:
            max_steps (int): Maximum number of optimization steps.
            quick_mode (bool): If True, reduces the number of optimization steps and speeds up the process.
            max_pairs (int): Maximum number of image pairs to consider from the viewgraph. Only used if quick_mode is True.
        """
        time_start = time.time()

        if quick_mode and len(self.viewgraph) > max_pairs:
            print(
                f"Quick mode ON. Randomly sampling viewgraph from {len(self.viewgraph):,} to {max_pairs:,} pairs before every iteration."
            )

        # Loop
        bar = tqdm(range(max_steps), desc="Adjusting poses and intrinsics")
        for _ in bar:
            # initialize loss
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                loss = 0.0
                self.optimizer.zero_grad()

                # Loop over pairs
                if len(self.viewgraph) > max_pairs and quick_mode:
                    sampled_indices = torch.randperm(len(self.viewgraph))[:max_pairs]
                    sampled_viewgraph = [self.viewgraph[i] for i in sampled_indices]
                else:
                    sampled_viewgraph = self.viewgraph

                # To get compatibility with LM, this should be the forward pass
                # of a nn.Module class. Also need to store parameter internally.
                # TODO: Batch this part to speed up, return (N,) DT values.
                # Then I can mean or pass to LM
                for pair in sampled_viewgraph:
                    i, j = pair
                    image_i = self.images[i]
                    image_j = self.images[j]
                    K_i = self.intrinsics[image_i["cam_id"]]
                    K_j = self.intrinsics[image_j["cam_id"]]

                    # project edges 1->2
                    edges_12 = reproject_2D_2D(
                        xy0=image_i["edges"][None],
                        depthmap0=image_i["depth"][None],
                        P0=image_i["P"].projection_matrix()[None],
                        P1=image_j["P"].projection_matrix()[None],
                        K0=K_i.intrinsic_matrix()[None],
                        K1=K_j.intrinsic_matrix()[None],
                        img1_shape=image_j["image"].shape[-2:],
                    )

                    # project edges 2->1
                    edges_21 = reproject_2D_2D(
                        xy0=image_j["edges"][None],
                        depthmap0=image_j["depth"][None],
                        P0=image_j["P"].projection_matrix()[None],
                        P1=image_i["P"].projection_matrix()[None],
                        K0=K_j.intrinsic_matrix()[None],
                        K1=K_i.intrinsic_matrix()[None],
                        img1_shape=image_i["image"].shape[-2:],
                    )

                    # Remove batch dimension before sampling
                    edges_12 = edges_12.squeeze(0)  # (N, 2)
                    edges_21 = edges_21.squeeze(0)  # (N, 2)

                    # Filter out NaN values
                    if edges_12.numel() > 0:
                        valid_12 = ~torch.isnan(edges_12).any(dim=1)
                    if edges_21.numel() > 0:
                        valid_21 = ~torch.isnan(edges_21).any(dim=1)
                    edges_12 = edges_12[valid_12]
                    edges_21 = edges_21[valid_21]

                    # compute loss
                    edge_loss_12 = sample_distance_field(
                        image_j["dt_field"], edges_12, device=self.device
                    )

                    edge_loss_21 = sample_distance_field(
                        image_i["dt_field"], edges_21, device=self.device
                    )

                    edge_loss = edge_loss_12.mean() + edge_loss_21.mean()
                    loss += edge_loss
                    # self.pair_losses.append((i, j, edge_loss.item()))

            loss /= len(sampled_viewgraph)
            self.loss_list.append(loss.item())

            # Backpropagate and update (for all poses and intrinsics)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bar.set_postfix(
                loss=f"{loss.item():.3f}",
            )

        self.timings["total_optimization"] += time.time() - time_start
        self.print_summary()

    def print_summary(self, w=None):
        # Column widths
        key_width = 30
        val_width = 10
        perc_width = 8

        # Total line width
        if w is None:
            w = key_width + val_width + perc_width + 4

        print("\n" + "=" * w)
        print(f"{'Summary':^{w}}")
        print("-" * w)

        # Compute total
        self.timings["total"] = (
            self.timings["total_loading"] + self.timings["total_optimization"]
        )

        # Header row
        print(f"{'Stage':<{key_width}}{'Time (s)':>{val_width}}{'%':>{perc_width+2}}")
        print("-" * w)

        # Per-key entries
        for key, value in self.timings.items():
            if (key in ["total", "total_loading"]) or (
                key == "total_optimization" and value == 0
            ):
                continue
            perc = (value / self.timings["total"]) * 100
            print(
                f"{key:<{key_width}}{value:>{val_width}.2f}  {perc:>{perc_width}.1f}%"
            )

        print("-" * w)
        print(f"{'Total':<{key_width}}{self.timings['total']:>{val_width}.2f}")

        # Loss summary
        if len(self.loss_list) > 0:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss

            print("-" * w)
            print(
                f"{'Initial loss:':<{key_width}}{initial_loss:>{val_width + perc_width + 4}.6f}"
            )
            print(
                f"{'Final loss:':<{key_width}}{final_loss:>{val_width + perc_width + 4}.6f}"
            )
            print(
                f"{'Loss reduction:':<{key_width}}{delta:>{val_width + perc_width + 4}.6f}"
            )
            steps = 0 if len(self.loss_list) == 0 else len(self.loss_list)
            print(
                f"{'Total steps:':<{key_width}}{steps:>{val_width + perc_width + 4}d}"
            )

        print("=" * w)

    def fix_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def load_optimizer(self, optim_name, params):
        optim_name = optim_name.lower()
        self.use_pypose = False

        # Build parameter groups only for parameters that exist
        param_groups = []
        if "k" in params:
            param_groups.append({"params": params["k"], "lr": self.lr * 0.5})
        if "t" in params:
            param_groups.append({"params": params["t"], "lr": self.lr})
        if "q" in params:
            param_groups.append({"params": params["q"], "lr": self.lr * 0.1})

        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        elif optim_name == "lm":
            try:
                import pypose as pp

                class ParamWrapper(nn.Module):
                    def __init__(self, params_dict):
                        super().__init__()
                        # Flatten all params from dict
                        all_params = []
                        for param_list in params_dict.values():
                            all_params.extend(param_list)
                        self.params = nn.ParameterList(all_params)

                params_wrapper = ParamWrapper(params)

                optimizer = pp.optim.LevenbergMarquardt(model=params_wrapper)
                self.scheduler = pp.optim.scheduler.StopOnPlateau(
                    optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True
                )

                self.use_pypose = True
                print("Using PyPose Levenberg-Marquardt optimizer.")

            except ImportError:
                print("PyPose not found. Falling back to AdamW.")
                optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        else:
            print(f"Optimizer {optim_name} not recognized. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        return optimizer

    def process_camera(self, camera, load_with_pad=False):
        # Convert a single pycolmap.Camera to torch tensor
        cam_id = camera.camera_id
        model = camera.model.name
        params = camera.params
        width = camera.width
        height = camera.height

        if model == "SIMPLE_PINHOLE":  # or model == "SIMPLE_RADIAL":
            f = params[0]
            cx, cy = params[1], params[2]

        elif model == "PINHOLE":  # or model == "RADIAL":
            f = torch.tensor([params[0], params[1]], dtype=torch.float32)
            cx, cy = params[2], params[3]

        else:
            raise NotImplementedError(f"Camera model {model} not supported.")

        # Account for padding when making square
        max_dim = max(width, height)
        pad_x = (max_dim - width) // 2 if load_with_pad else 0
        pad_y = (max_dim - height) // 2 if load_with_pad else 0

        # Scale factor after resize
        scale = self.images_size / max_dim

        # Apply padding shift + scale
        f = f * scale
        cx = (cx + pad_x) * scale
        cy = (cy + pad_y) * scale

        params = torch.cat([f, torch.tensor([cx, cy], dtype=torch.float32)], dim=0)
        return cam_id, model, params

    def process_pose(self, image):
        # Convert a single pycolmap.Image to torch tensor
        # COLMAP's cam_from_world is already R_cw (world-to-camera rotation)
        R = torch.tensor(image.cam_from_world.rotation.matrix(), dtype=torch.float32)
        t = torch.tensor(
            image.cam_from_world.translation, dtype=torch.float32
        ).unsqueeze(1)

        return R, t, image.camera_id

    def read_cameras_from_reconstruction(
        self,
        reconstruction,
        images,
        single_camera_per_folder=False,
        load_with_pad=False,
    ):

        intrinsics = {}

        # Read cameras intrinsics
        if single_camera_per_folder:
            # Reading cameras from images (to handle multiple images with same camera)
            for image in reconstruction.images.values():
                _, model, new_params = self.process_camera(image.camera, load_with_pad)
                cam_id = image.name.split("/")[
                    0
                ]  # assuming image names are like "cam_id/image_name"

                # I want to stack params of same cam_id to then averaged them
                if cam_id not in intrinsics:
                    intrinsics[cam_id] = {
                        "cam_id": cam_id,
                        "model": model,
                        "parameters": [new_params],  # Changed: store as list initially
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
            for cam in reconstruction.cameras.values():
                cam_id, model, new_params = self.process_camera(cam)
                intrinsics[cam.camera_id] = {
                    "cam_id": cam_id,
                    "model": model,
                    "parameters": new_params.to(self.device),
                }
        # Sort dict by keys
        intrinsics = dict(sorted(intrinsics.items()))

        # Convert to Camera objects
        for cam_id in intrinsics.keys():
            intrinsics[cam_id] = Camera(
                **intrinsics[cam_id], grad=self.grad_k, device=self.device
            )

        # Read poses from images
        for image in reconstruction.images.values():
            R, t, cam_id = self.process_pose(image)
            pose = Pose(R=R, t=t, grad_q=self.grad_q, device=self.device)

            if single_camera_per_folder:
                cam_id = image.name.split("/")[0]
            else:
                cam_id = image.camera_id

            images[image.name].update({"P": pose, "cam_id": cam_id})

        return images, intrinsics

    def build_reconstruction(
        self, output_path="optimized_reconstruction", save_points=False
    ):
        """
        Create a pycolmap.Reconstruction from images and intrinsics dictionaries.

        Args:
            images: dict with image_name -> {P: Pose, cam_id: int, scale: float, ...}
            intrinsics: dict with cam_id -> Camera
            output_path: path to save the reconstruction
            save_points: whether to save 3D points (empty here)
        """
        # Create empty reconstruction
        reconstruction = pycolmap.Reconstruction()

        # 1. Add cameras - we need to handle different scales per image
        # Group images by camera to find the appropriate scale
        camera_scales = {}
        for image_name, image_data in self.images.items():
            cam_id = image_data["cam_id"]
            scale = image_data.get("scale", 1.0)

            if cam_id not in camera_scales:
                camera_scales[cam_id] = []
            camera_scales[cam_id].append(scale)

        # Use median scale for each camera
        for cam_id in camera_scales:
            camera_scales[cam_id] = np.median(camera_scales[cam_id])

        for cam_id, camera in self.intrinsics.items():
            # Get camera parameters as numpy array
            params = camera.params.detach().cpu().numpy()

            # Get scale for this camera
            scale = camera_scales.get(cam_id, 1.0)

            # Apply inverse scaling to focal lengths (scale back to original)
            if camera.model == "PINHOLE":
                params = params.copy()
                params[0] /= scale  # fx
                params[1] /= scale  # fy
                params[2] /= scale  # cx
                params[3] /= scale  # cy
                model = pycolmap.CameraModelId.PINHOLE
            elif camera.model == "SIMPLE_PINHOLE":
                params = params.copy()
                params[0] /= scale  # f
                params[1] /= scale  # cx
                params[2] /= scale  # cy
                model = pycolmap.CameraModelId.SIMPLE_PINHOLE
            else:
                raise ValueError(f"Unsupported camera model: {camera.model}")

            # Get image dimensions from first image with this cam_id
            sample_image = next(
                (img for img in self.images.values() if img["cam_id"] == cam_id), None
            )

            if sample_image is None:
                print(f"Warning: No images found for camera {cam_id}, skipping...")
                continue

            height, width = sample_image["depth"].shape[-2:]

            # Scale image dimensions back to original
            width = int(width / scale)
            height = int(height / scale)

            # Convert cam_id to int for COLMAP
            cam_id_int = int(cam_id) if isinstance(cam_id, str) else cam_id

            # Create and register camera
            cam = pycolmap.Camera(
                model=model,
                width=width,
                height=height,
                params=params,
                camera_id=cam_id_int,
            )
            reconstruction.add_camera(cam)

        # 2. Add images (poses)
        for image_id, (image_name, image_data) in enumerate(
            self.images.items(), start=1
        ):
            pose = image_data["P"]
            cam_id = image_data["cam_id"]
            scale = image_data.get("scale", 1.0)

            # Convert cam_id to int for COLMAP
            cam_id_int = int(cam_id) if isinstance(cam_id, str) else cam_id

            # Get rotation matrix and translation
            q = pose.q.detach().cpu().numpy()
            t = pose.t.detach().cpu().numpy().squeeze()

            # Apply inverse scaling to translation (scale back to original)
            t = t / scale

            # use xyzw -> wxyz
            # q = np.roll(q, shift=3)

            # Create image
            img = pycolmap.Image(
                id=image_id,
                name=image_name,
                camera_id=cam_id_int,
                cam_from_world=pycolmap.Rigid3d(
                    rotation=pycolmap.Rotation3d(q), translation=t
                ),
            )
            reconstruction.add_image(img)

        # 3. Points3D - empty for now
        if save_points:
            print("Saving with empty Points3D...")
            # TODO:similar to VGGT, unproject depth maps to create points3D
            pass
        else:
            print("Saving without Points3D...")

        # 4. Save reconstruction
        print(f"Cameras: {len(reconstruction.cameras)}")
        print(f"Images: {len(reconstruction.images)}")
        print(f"Points3D: {len(reconstruction.points3D)}")

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            reconstruction.write_text(output_path)
            print(f"Reconstruction saved to: {output_path}")

        return reconstruction


if __name__ == "__main__":

    adjuster = Adjuster(
        reconstruction="/home/mattia/Desktop/Repos/vggt/wrapper_output/vienna_state_opera/sparse",
        images_path="/home/mattia/Desktop/datasets/mydataset/data/vienna_state_opera/frames",
        depths_path="/home/mattia/Desktop/Repos/vggt/wrapper_output/vienna_state_opera/sparse/depth_maps",
        single_camera_per_folder=True,
        load_with_pad=False,
        lr=1e-3,
        grad_q=True,
        grad_t=True,
        grad_k=True,
        optim="adamw",
    )

    out = adjuster(
        quick_mode=True,
        max_pairs=512,
        max_steps=30,
    )
