# This script should contain the pose, intrinsics and depth optimizer.
# It should work with PyTorch and be compatible with CUDA if available.
# It should works usign edges and photometric losses.

# I might want to use pypose, kornia and/or pytorch3d for this.
import os
import time
import torch
import torch.nn as nn

import warnings

import warnings

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
from adjuster.helpers import (
    build_view_graph_from_frustums,
    find_images,
    load_and_preprocess_images_square,
    load_and_preprocess_depths_square,
)

from adjuster.extractors import CannyEdgeDetector


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
        self, detector="canny", device="cuda", max_workers=-1, detector_params={}
    ):
        super().__init__()

        self.max_workers = os.cpu_count() if max_workers < 0 else max_workers
        self.images_size = 518  # square size to which images are resized

        # Edge extractor
        if detector == "canny":
            self.edge_extractor = CannyEdgeDetector(
                low_threshold=detector_params.get("low_threshold", 0.15),
                high_threshold=detector_params.get("high_threshold", 0.25),
                hysteresis=detector_params.get("hysteresis", True),
                kernel_size=detector_params.get("kernel_size", 9),
                sigma=detector_params.get("sigma", 7.0),
                device=device,
            )

    def forward(
        self,
        images_path,
        depths_path,
        reconstruction,
        max_steps=5_000,
        lr=1e-4,
    ):
        """
        Main optimization loop.
        Args:
            images_path (str): Path to images.
            depth_maps (torch.Tensor): Initial depth maps of shape (B, 1, H, W).
            reconstruction (pycolmap.Reconstruction): Initial COLMAP reconstruction.
            max_steps (int): Maximum number of optimization steps.
            lr (float): Learning rate for the optimizer.
        """
        timings = {}
        time_start = time.time()
        # assert use_photo_loss or use_edge_loss, "At least one loss must be used."

        # View graph
        s_time = time.time()
        recon, viewgraph = build_view_graph_from_frustums(
            reconstruction,
            z_near_default=0.1,
            z_far_default=5.0,
            max_view_angle_deg=70.0,
            distance_factor=2,
            verbose=False,
        )
        timings["viewgraph"] = time.time() - s_time

        # Load Images as dict {image_name: image_tensor}
        s_time = time.time()
        image_path_list = find_images(images_path)
        images = load_and_preprocess_images_square(
            image_path_list,
            images_path,
            target_size=self.images_size,
            max_workers=self.max_workers,
        )
        timings["load_images"] = time.time() - s_time

        # Extract edges
        s_time = time.time()
        for image_name in tqdm(images.keys(), desc="Extracting edges"):
            img_tensor = (
                images[image_name]["image"].unsqueeze(0).to(self.edge_extractor.device)
            )
            edges = self.edge_extractor(img_tensor)
            images[image_name].update({"edges": edges.squeeze().cpu()})
        timings["extract_edges"] = time.time() - s_time

        # Load poses and intrinsics
        s_time = time.time()
        images, intrinsics = self.read_cameras_from_reconstruction(recon, images)
        timings["load_poses_and_intrinsics"] = time.time() - s_time

        # Load depth maps
        s_time = time.time()
        images = load_and_preprocess_depths_square(
            depths_path,
            images,
            target_size=self.images_size,
            max_workers=self.max_workers,
        )
        timings["load_depth_maps"] = time.time() - s_time

        timings["total"] = time.time() - time_start
        self.print_timings(timings)
        return images, viewgraph, intrinsics

        if True:
            # # Optimizer
            # NOTE: VGGT Translation is prone to error. An alternating round of optimization only for it? or bilirnear cost function

            # optimizer = torch.optim.Adam(self.parameters(), lr=lr)

            # # Loop
            # bar = tqdm(range(max_steps), desc="Adjusting poses and depth maps")
            # for step in bar:
            #     # initialize loss
            #     loss = 0
            #     optimizer.zero_grad()

            #     # Loop over pairs
            #     for pair in pairs:
            #         image_1, image_2, _ = pair

            #         # compute losses
            #         if use_photo_loss:
            #             # self.project_image(...) / self.project_patches
            #             ...
            #             photo_loss = ...
            #             loss += photo_loss

            #         if use_edge_loss:
            #             # project edges 1->2
            #             edges_12 = self.project_edges(
            #                 images_edges[image_1],
            #                 poses[image_1],
            #                 poses[image_2],
            #                 intrinsics[image_1],
            #                 intrinsics[image_2],
            #                 depth_maps[image_1],
            #             )

            #             # project edges 2->1
            #             edges_21 = self.project_edges(
            #                 images_edges[image_2],
            #                 poses[image_2],
            #                 poses[image_1],
            #                 intrinsics[image_2],
            #                 intrinsics[image_1],
            #                 depth_maps[image_2],
            #             )

            #             ...
            #             edge_loss = ...
            #             loss += edge_loss

            #     # Backpropagate and update (for all poses, intrinsics, depth maps)
            #     loss.backward()
            #     optimizer.step()

            #     bar.set_postfix(
            #         loss=f"{loss.item():.6f}",
            #         photo=f"{photo_loss.item():.6f}",
            #         edge=f"{edge_loss.item():.6f}",
            #     )

            # return depth_maps, reconstruction
            pass

    def print_timings(self, timings):
        print("=" * 40)
        print("Timings:")
        print("-" * 40)

        # Define column widths
        key_width = 30
        val_width = 10
        perc_width = 8

        for key, value in timings.items():
            if key == "total":
                continue
            print(
                f"{key:<{key_width}}{value:>{val_width}.2f} s {((value / timings['total']) * 100):>{perc_width}.1f}%"
            )

        print("-" * 40)
        print(f"{'total':<{key_width}}{timings['total']:>{val_width}.2f} s")
        print("=" * 40)

    def process_camera(self, camera):
        # Convert a single pycolmap.Camera to torch tensor
        model = camera.model.name
        params = camera.params
        width = camera.width
        height = camera.height

        if model == "SIMPLE_PINHOLE" or model == "SIMPLE_RADIAL":
            fx = params[0]
            fy = params[0]
            cx = params[1]
            cy = params[2]
        elif model == "PINHOLE" or model == "RADIAL":
            fx = params[0]
            fy = params[1]
            cx = params[2]
            cy = params[3]
        else:
            raise NotImplementedError(f"Camera model {model} not supported.")

        # Account for padding when making square
        max_dim = max(width, height)
        pad_x = (max_dim - width) // 2
        pad_y = (max_dim - height) // 2

        # Scale factor after resize
        scale = self.images_size / max_dim

        # Apply padding shift + scale
        fx = fx * scale
        fy = fy * scale
        cx = (cx + pad_x) * scale
        cy = (cy + pad_y) * scale

        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        return K

    def process_pose(self, image):
        # Convert a single pycolmap.Image to torch tensor
        # COLMAP's cam_from_world is already R_cw (world-to-camera rotation)
        R = torch.tensor(image.cam_from_world.rotation.matrix(), dtype=torch.float32)
        t = torch.tensor(
            image.cam_from_world.translation, dtype=torch.float32
        ).unsqueeze(1)

        # Build extrinsic matrix [R|t] (world-to-camera)
        # This is correct for projection: x_cam = R * x_world + t
        extrinsics = torch.cat([R, t], dim=1)  # 3x4 matrix

        # Only build 4x4 if you need homogeneous coordinates
        P = torch.cat(
            [
                extrinsics,
                torch.tensor([[0, 0, 0, 1]], dtype=torch.float32),
            ],
            dim=0,
        )

        return P, image.camera_id

    def read_cameras_from_reconstruction(self, reconstruction, images):
        # Read cameras intrinsics
        intrinsics = {}
        for cam in reconstruction.cameras.values():
            intrinsics[cam.camera_id] = self.process_camera(cam)

        for image in reconstruction.images.values():
            pose, cam_id = self.process_pose(image)
            images[image.name].update({"P": pose, "cam_id": cam_id})

        return images, intrinsics

    def update_reconstruction(self):
        # Update pycolmap.Reconstruction with new poses and intrinsics from torch tensors
        pass
