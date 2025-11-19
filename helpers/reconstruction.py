import os
import torch
import pycolmap
import numpy as np
from sklearn.cluster import DBSCAN


@torch.no_grad()
def dbscan_filter(reconstruction, eps=0.5, min_samples=20):
    """
    Filter 3D points in reconstruction using DBSCAN clustering.
    Keeps only the largest cluster to remove outliers.

    Args:
        reconstruction: pycolmap.Reconstruction object
        eps: Maximum distance between two samples for DBSCAN
        min_samples: Minimum number of samples in a neighborhood for DBSCAN

    Returns:
        pycolmap.Reconstruction: Filtered reconstruction
    """

    if len(reconstruction.points3D) == 0:
        return reconstruction

    # Extract 3D point coordinates
    point_ids = list(reconstruction.point3D_ids())
    xyz = np.array([reconstruction.point3D(p_id).xyz for p_id in point_ids])

    # Run DBSCAN with fallback for memory errors
    labels = None
    n_jobs = 4
    while n_jobs >= 1 and labels is None:
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(
                xyz
            )
            labels = clustering.labels_
        except MemoryError:
            n_jobs //= 2
        except Exception as e:
            print(f"DBSCAN failed: {e}")
            return reconstruction

    # Find largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    non_noise_indices = np.where(unique_labels != -1)

    if len(counts[non_noise_indices]) == 0:
        print("Warning: DBSCAN found no clusters. Skipping filtering.")
        return reconstruction

    main_cluster_label = unique_labels[non_noise_indices][
        np.argmax(counts[non_noise_indices])
    ]

    # Get indices of points in main cluster
    cluster_indices = np.where(labels == main_cluster_label)[0]
    ids_to_keep = set([point_ids[i] for i in cluster_indices])

    print(
        f"DBSCAN: Keeping largest cluster with {len(ids_to_keep):,} / {len(point_ids):,} points"
    )

    # Create new reconstruction with only kept points
    filtered_reconstruction = pycolmap.Reconstruction()

    # Copy cameras
    for camera_id in reconstruction.cameras.keys():
        filtered_reconstruction.add_camera(reconstruction.cameras[camera_id])

    # Recreate images with new camera references
    for image_id in reconstruction.images.keys():
        old_image = reconstruction.images[image_id]
        new_image = pycolmap.Image(
            id=old_image.image_id,
            name=old_image.name,
            camera_id=old_image.camera_id,
            cam_from_world=old_image.cam_from_world,
        )
        filtered_reconstruction.add_image(new_image)

    # Copy only kept points with fresh empty tracks
    for p_id in ids_to_keep:
        point3d = reconstruction.point3D(p_id)
        # Create empty track instead of copying old one
        empty_track = pycolmap.Track()
        filtered_reconstruction.add_point3D(point3d.xyz, empty_track, point3d.color)

    return filtered_reconstruction


@torch.no_grad()
def build_reconstruction(
    self,
    output_path="optimized_reconstruction",
    save_points=True,
    verbose=False,
    max_points_per_image=100_000,
    final_dbscan_filtering=False,
    dbscan_eps=0.05,
    dbscan_min_samples=5,
):
    """
    Create a pycolmap.Reconstruction from images and intrinsics dictionaries.

    Args:
        output_path: path to save the reconstruction
        save_points: whether to save 3D points from depth unprojection
        max_points_per_image: maximum number of 3D points per image (default: 100_000)
        dbscan_eps: epsilon for DBSCAN clustering
        dbscan_min_samples: min samples for DBSCAN clustering
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

    for cam_id in self.intrinsics.keys:
        # Get camera parameters as numpy array
        model, params = self.intrinsics.get_camera_parameters(cam_id)
        params = params.detach().cpu().numpy()

        # Get scale for this camera
        scale = camera_scales.get(cam_id, 1.0)

        # Apply inverse scaling to focal lengths (scale back to original)
        if model == "PINHOLE":
            params = params.copy()
            params[0] /= scale  # fx
            params[1] /= scale  # fy
            params[2] /= scale  # cx
            params[3] /= scale  # cy
            model = pycolmap.CameraModelId.PINHOLE

        elif model == "SIMPLE_PINHOLE":
            params = params.copy()
            params[0] /= scale  # f
            params[1] /= scale  # cx
            params[2] /= scale  # cy
            model = pycolmap.CameraModelId.SIMPLE_PINHOLE
        else:
            raise ValueError(f"Unsupported camera model: {model}")

        # Get image dimensions from first image with this cam_id
        sample_image = next(
            (img for img in self.images.values() if img["cam_id"] == cam_id), None
        )

        if sample_image is None:
            print(f"Warning: No images found for camera {cam_id}, skipping...")
            continue

        height, width = sample_image["hw"]

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
    for image_id, (image_name, image_data) in enumerate(self.images.items(), start=1):
        # pose = self.poses.get_projection_matrix([image_name])
        cam_id = image_data["cam_id"]
        scale = image_data.get("scale", 1.0)

        # Convert cam_id to int for COLMAP
        cam_id_int = int(cam_id) if isinstance(cam_id, str) else cam_id

        # Get rotation matrix and translation
        q, t = self.poses.get_image_qt([image_name])
        q = q.detach().cpu().numpy()
        t = t.detach().cpu().numpy()

        # Apply inverse scaling to translation (scale back to original)
        t = t / scale

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

    # 3. Add Points3D from depth unprojection using fresh computation
    if save_points:
        if verbose:
            print("Unprojecting depth maps to 3D points...")

        # Compute fresh 3D world coordinates
        self._unproject_edges_to_3D()

        total_points = 0

        for image_id, (image_name, image_data) in enumerate(
            self.images.items(), start=1
        ):
            cam_id = image_data["cam_id"]
            scale = image_data.get("scale", 1.0)

            # Get unprojected 3D points in world coordinates (at scaled resolution)
            edges_3D = image_data.get("edges_3D", None)  # (N, 3)
            pad_mask = image_data.get("pad_mask", None)  # (N,)

            if edges_3D is None or pad_mask is None:
                if verbose:
                    print(f"No edges_3D or pad_mask for {image_name}, skipping...")
                continue

            # Convert to numpy
            edges_3D_np = edges_3D.detach().cpu().numpy()  # (N, 3)
            pad_mask_np = pad_mask.detach().cpu().numpy()  # (N,)

            # Filter by pad mask (only valid edges, ignore padded entries)
            valid_mask = pad_mask_np > 0
            valid_3D = edges_3D_np[valid_mask]  # (M, 3)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_3D) == 0:
                if verbose:
                    print(f"No valid edges for {image_name}")
                continue

            # Scale points back to original resolution
            # points_3D is in downsampled world space, multiply by scale to expand to original
            valid_3D = valid_3D / scale

            # Sample uniformly up to max_points_per_image
            num_valid = len(valid_3D)
            if num_valid > max_points_per_image:
                sample_idx = np.random.choice(
                    num_valid, size=max_points_per_image, replace=False
                )
                valid_3D = valid_3D[sample_idx]
                valid_indices = valid_indices[sample_idx]

            # Get RGB values from original image
            if "image" in image_data:
                image = image_data["image"].detach().cpu().numpy()  # (3, H, W)
                edges_padded = (
                    image_data["edges_padded"].detach().cpu().numpy()
                )  # (N, 2)

                # Get coordinates of valid edges
                valid_edges = edges_padded[valid_indices]
                y_coords_int = valid_edges[:, 1].astype(np.int32)
                x_coords_int = valid_edges[:, 0].astype(np.int32)

                # Clamp to valid range (at scaled resolution)
                y_coords_int = np.clip(y_coords_int, 0, image.shape[1] - 1)
                x_coords_int = np.clip(x_coords_int, 0, image.shape[2] - 1)

                rgb = image[:, y_coords_int, x_coords_int]  # (3, M)
                rgb = (rgb * 255).astype(np.uint8).T  # (M, 3)
            else:
                # Default to black if no image available
                rgb = np.full((len(valid_3D), 3), 0, dtype=np.uint8)

            # Add points to reconstruction
            for pt_world, rgb_val in zip(valid_3D, rgb):
                point3D_id = reconstruction.add_point3D(
                    pt_world, pycolmap.Track(), rgb_val
                )
                track = reconstruction.point3D(point3D_id).track
                track.add_element(image_id, int(0))  # dummy keypoint index

            total_points += len(valid_3D)
            if verbose:
                print(f"Added {len(valid_3D)} points from {image_name}")

        if verbose:
            print(f"Total points added: {total_points:,}")

    # 4. DBSCAN filtering
    if save_points and final_dbscan_filtering:
        if verbose:
            print("Running DBSCAN filtering...")
        reconstruction = dbscan_filter(
            reconstruction, eps=dbscan_eps, min_samples=dbscan_min_samples
        )

    # 5. Save reconstruction
    if verbose:
        print(f"Cameras: {len(reconstruction.cameras)}")
        print(f"Images: {len(reconstruction.images)}")
        print(f"Points3D: {len(reconstruction.points3D):,}")

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        reconstruction.write_text(output_path)
        if verbose:
            print(f"Reconstruction saved to: {output_path}")

    return reconstruction
