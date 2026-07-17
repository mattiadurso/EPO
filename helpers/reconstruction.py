"""Build / filter / save ``pycolmap.Reconstruction`` objects.

Provides :func:`build_reconstruction`, which assembles a full COLMAP
reconstruction from an EPO instance (cameras, poses and the per-image
unprojected edge points), and :func:`dbscan_filter`, an outlier-removal
pass that keeps only the largest DBSCAN cluster of 3D points.

Updated for pycolmap 4.x: poses now live on ``Frame`` objects (which
reference a ``Rig``), not directly on ``Image``. Each image's data must
be associated with its frame via ``Frame.add_data_id`` or the
reconstruction will reject it (``frame.HasDataId(image.DataId())``).
"""

import logging
import os
import warnings

import numpy as np
import pycolmap
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def dbscan_filter(reconstruction, eps=0.5, min_samples=20, verbose: bool = False):
    """Filter 3D points in reconstruction using DBSCAN clustering.
    Keeps only the largest cluster to remove outliers.

    Args:
        reconstruction: pycolmap.Reconstruction object
        eps: Maximum distance between two samples for DBSCAN
        min_samples: Minimum number of samples in a neighborhood for DBSCAN
        verbose: If True, log cluster statistics.

    Returns:
        pycolmap.Reconstruction: Filtered reconstruction (the input is
        returned unchanged when DBSCAN fails or finds no clusters).
    """
    # Lazy import — keeps sklearn off the import graph until DBSCAN filtering
    # is actually requested (it's an opt-in post-processing step).
    from sklearn.cluster import DBSCAN

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
            warnings.warn(f"DBSCAN failed: {e}; returning unfiltered reconstruction.")
            return reconstruction

    # Find largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    non_noise_indices = np.where(unique_labels != -1)

    if len(counts[non_noise_indices]) == 0:
        warnings.warn("DBSCAN found no clusters; returning unfiltered reconstruction.")
        return reconstruction

    main_cluster_label = unique_labels[non_noise_indices][
        np.argmax(counts[non_noise_indices])
    ]

    # Get indices of points in main cluster
    cluster_indices = np.where(labels == main_cluster_label)[0]
    ids_to_keep = set([point_ids[i] for i in cluster_indices])

    if verbose:
        logger.info(
            f"DBSCAN: keeping largest cluster with "
            f"{len(ids_to_keep):,} / {len(point_ids):,} points"
        )

    # Create new reconstruction with only kept points
    filtered_reconstruction = pycolmap.Reconstruction()

    # Copy cameras
    for camera_id in reconstruction.cameras.keys():
        filtered_reconstruction.add_camera(reconstruction.cameras[camera_id])

    # Copy rigs (4.x: rigs must exist before frames/images that reference them)
    for rig_id in reconstruction.rigs.keys():
        filtered_reconstruction.add_rig(reconstruction.rigs[rig_id])

    # Recreate frames + images with the 4.x API. The pose lives on the
    # frame; the image's data_id must be registered on the frame before
    # add_image, or the reconstruction rejects it.
    for image_id in reconstruction.images.keys():
        old_image = reconstruction.images[image_id]
        old_frame = reconstruction.frames[old_image.frame_id]
        camera = reconstruction.cameras[old_image.camera_id]

        new_frame = pycolmap.Frame()
        new_frame.frame_id = old_frame.frame_id
        new_frame.rig_id = old_frame.rig_id
        new_frame.rig = filtered_reconstruction.rigs[old_frame.rig_id]
        new_frame.set_cam_from_world(old_image.camera_id, old_image.cam_from_world())
        new_frame.add_data_id(pycolmap.data_t(camera.sensor_id, old_image.image_id))
        filtered_reconstruction.add_frame(new_frame)

        new_image = pycolmap.Image(
            image_id=old_image.image_id,
            name=old_image.name,
            camera_id=old_image.camera_id,
            frame_id=new_frame.frame_id,
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
    epo,
    output_path="optimized_reconstruction",
    save_points=True,
    verbose=False,
    max_points_per_image=100_000,
    final_dbscan_filtering=False,
    dbscan_eps=0.05,
    dbscan_min_samples=5,
    bin=False,
):
    """Create a pycolmap.Reconstruction from images and intrinsics dictionaries.

    Args:
        epo: epo instance with images, poses, and intrinsics
        output_path: path to save the reconstruction
        save_points: whether to save 3D points from depth unprojection
        verbose: if True, print progress information while assembling the reconstruction
        max_points_per_image: maximum number of 3D points per image (default: 100_000)
        final_dbscan_filtering: if True, run DBSCAN outlier removal on the merged 3D
            points before writing the reconstruction
        dbscan_eps: epsilon for DBSCAN clustering
        dbscan_min_samples: min samples for DBSCAN clustering
        bin: if True, write COLMAP files in binary ``.bin`` format; otherwise ``.txt``
    """
    # free cache to avoid OOM while saving
    torch.cuda.empty_cache()

    # Create empty reconstruction
    reconstruction = pycolmap.Reconstruction()

    # 1. Add cameras - we need to handle different scales per image
    # Group images by camera to find the appropriate scale
    camera_scales = {}
    unique_cam_ids = set()

    for _image_name, image_data in epo.images.items():
        cam_id = image_data["cam_id"]
        scale = image_data.get("scale", 1.0)
        unique_cam_ids.add(cam_id)

        if cam_id not in camera_scales:
            camera_scales[cam_id] = []
        camera_scales[cam_id].append(scale)

    # Use median scale for each camera
    for cam_id in camera_scales:
        camera_scales[cam_id] = np.median(camera_scales[cam_id])

    for idx, cam_id in enumerate(
        unique_cam_ids
    ):  # Fixed: iterate over unique camera IDs found in images
        # Get camera parameters as numpy array
        model, params = epo.intrinsics.get_camera_parameters(cam_id)
        params = params.detach().cpu().float().numpy()

        # Get scale for this camera
        scale = camera_scales.get(cam_id, 1.0)

        # Apply inverse scaling to focal lengths (scale back to original)
        params = params.copy()
        params[0] /= scale  # f
        params[1] /= scale  # cx
        params[2] /= scale  # cy
        model = pycolmap.CameraModelId.SIMPLE_PINHOLE

        # Get image dimensions from first image with this cam_id
        sample_image = next(
            (img for img in epo.images.values() if img["cam_id"] == cam_id), None
        )

        if sample_image is None:
            warnings.warn(f"No images found for camera {cam_id}, skipping.")
            continue

        height, width = sample_image["hw"]

        # Scale image dimensions back to original
        width = int(width / scale)
        height = int(height / scale)

        # Convert cam_id to int for COLMAP
        if isinstance(cam_id, str):
            cam_id_int = epo.intrinsics.image_to_tensor_idx[cam_id]
        elif cam_id is None:
            cam_id_int = idx + 1  # assign a new ID
        elif isinstance(cam_id, int):
            cam_id_int = cam_id
        else:
            raise ValueError(
                f"Unsupported cam_id type: {type(cam_id)}, value: {cam_id}"
            )

        # Create and register camera
        cam = pycolmap.Camera(
            model=model,
            width=width,
            height=height,
            params=params,
            camera_id=cam_id_int,
        )
        reconstruction.add_camera(cam)

    # 2. Build one rig per camera (trivial: sensor == rig origin).
    #    Do this once, before the image loop, so shared cameras don't
    #    create duplicate rigs.
    seen_rigs = set()
    for cam_id_int in set(
        epo.intrinsics.image_to_tensor_idx[d["cam_id"]] for d in epo.images.values()
    ):
        if cam_id_int in seen_rigs:
            continue
        camera = reconstruction.cameras[cam_id_int]
        rig = pycolmap.Rig()
        rig.rig_id = cam_id_int
        rig.add_ref_sensor(camera.sensor_id)  # sensor_t; ref sensor == rig origin
        reconstruction.add_rig(rig)
        seen_rigs.add(cam_id_int)

    # 3. Add images + frames. In pycolmap 4.x the pose lives on the
    #    Frame, and the image's data_id must be registered on the frame
    #    (Frame.add_data_id) before the image is added, otherwise
    #    reconstruction.add_image fails the
    #    `frame.HasDataId(image.DataId())` check.
    #
    #    We keep a stable image_id -> name mapping so the points/track
    #    step below references the exact same ids.
    image_id_to_name = {}
    for image_id, (image_name, image_data) in enumerate(epo.images.items(), start=1):
        cam_id = image_data["cam_id"]
        scale = image_data.get("scale", 1.0)

        # Convert cam_id to int for COLMAP
        cam_id_int = epo.intrinsics.image_to_tensor_idx[cam_id]
        camera = reconstruction.cameras[cam_id_int]

        # Get rotation matrix and translation (R is already orthonormal).
        # get_image_Rt takes a list and returns a batched result, so
        # reshape to the strict (3, 3) / (3,) that Rotation3d / Rigid3d
        # expect.
        R, t = epo.poses.get_image_Rt([image_name])
        R = R.detach().cpu().numpy().astype(np.float64).reshape(3, 3)
        t = t.detach().cpu().numpy().astype(np.float64).reshape(3)

        # Apply inverse scaling to translation (scale back to original)
        t = t / scale

        cam_from_world = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(R), translation=t
        )

        # Frame holds the pose. set_cam_from_world dereferences the
        # frame's rig pointer (frame.cc:62), so the live Rig object must
        # be attached before that call -- rig_id alone is not enough.
        frame = pycolmap.Frame()
        frame.frame_id = image_id
        frame.rig_id = cam_id_int
        frame.rig = reconstruction.rigs[cam_id_int]
        frame.set_cam_from_world(cam_id_int, cam_from_world)
        # Associate this image's data with the frame BEFORE add_image.
        frame.add_data_id(pycolmap.data_t(camera.sensor_id, image_id))
        reconstruction.add_frame(frame)

        # Image just associates name + camera + frame (no pose)
        img = pycolmap.Image(
            image_id=image_id,
            name=image_name,
            camera_id=cam_id_int,
            frame_id=frame.frame_id,
        )
        reconstruction.add_image(img)
        image_id_to_name[image_id] = image_name

    # 4. Add Points3D from depth unprojection using fresh computation.
    #    Reuse the exact image_id -> name mapping built above so track
    #    elements reference registered image ids.
    if save_points:
        if verbose:
            logger.info("Unprojecting depth maps to 3D points...")

        # Compute fresh 3D world coordinates
        epo.unproject_edges_to_3D()

        total_points = 0

        for _image_id, image_name in image_id_to_name.items():
            image_data = epo.images[image_name]
            cam_id = image_data["cam_id"]
            scale = image_data.get("scale", 1.0)

            # Get unprojected 3D points from edges_3D module
            points_3D = epo.edges_3D.get_parameters([image_name])  # (1, N, 3)
            points_3D = points_3D[0]  # (N, 3)

            # Get pad mask
            pad_mask = epo.pad_masks.get_parameters([image_name])  # (1, N)
            pad_mask = pad_mask[0]  # (N,)

            # Convert to numpy if needed
            if torch.is_tensor(pad_mask):
                pad_mask = pad_mask.detach().cpu().numpy()

            # Filter by pad mask (only valid edges, ignore padded entries)
            valid_mask = pad_mask > 0
            valid_3D = points_3D[valid_mask]  # (M, 3)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_3D) == 0:
                if verbose:
                    logger.warning(f"No valid edges for {image_name}")
                continue

            # Convert to numpy if needed
            if torch.is_tensor(valid_3D):
                valid_3D = valid_3D.detach().cpu().numpy()
            if torch.is_tensor(valid_indices):
                valid_indices = valid_indices.detach().cpu().numpy()

            # Scale points back to original resolution
            valid_3D = valid_3D / scale

            # Sample uniformly up to max_points_per_image
            num_valid = len(valid_3D)
            max_points_int = int(max_points_per_image)
            if num_valid > max_points_int:
                sample_idx = np.random.choice(
                    num_valid, size=max_points_int, replace=False
                )
                valid_3D = valid_3D[sample_idx]
                valid_indices = valid_indices[sample_idx]

            # Get RGB values from original image
            if "image" in image_data:
                image = image_data["image"].detach().cpu().numpy()  # (3, H, W)
                edges_padded = epo.edges_padded.get_parameters(
                    [image_name]
                )  # (1, N, 2)
                edges_padded = edges_padded[0].detach().cpu().numpy()  # (N, 2)

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

            # Add points to reconstruction.
            #
            # NOTE: we intentionally do NOT call track.add_element here.
            # A track element references (image_id, point2D_index), but
            # these images have no Point2D entries (we only export edge
            # points, not feature observations). Writing a track element
            # with a dangling 2D reference produces a model COLMAP can
            # serialize but NOT read back: on load it indexes
            # image.points2D[0] into an empty vector and raises
            # `vector::_M_range_check: __n (0) >= size (0)`.
            # Empty tracks are the correct representation for
            # observation-free points (mean track length 0).
            for pt_world, rgb_val in zip(valid_3D, rgb, strict=False):
                reconstruction.add_point3D(pt_world, pycolmap.Track(), rgb_val)

            total_points += len(valid_3D)
            if verbose:
                logger.info(f"Added {len(valid_3D)} points from {image_name}")

        if verbose:
            logger.info(f"Total points added: {total_points:,}")

    # 5. DBSCAN filtering
    if save_points and final_dbscan_filtering:
        if verbose:
            logger.info("Running DBSCAN filtering...")
        reconstruction = dbscan_filter(
            reconstruction,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            verbose=verbose,
        )

    # 6. Save reconstruction
    if verbose:
        logger.info(f"Cameras: {len(reconstruction.cameras)}")
        logger.info(f"Images: {len(reconstruction.images)}")
        logger.info(f"Points3D: {len(reconstruction.points3D):,}")

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        if bin:
            reconstruction.write_binary(output_path)
        else:
            reconstruction.write_text(output_path)
        if verbose:
            logger.info(f"Reconstruction saved to: {output_path}")

    if epo.images_not_in_viewgraph and verbose:
        logger.info(
            f"{len(epo.images_not_in_viewgraph)} images were not in the "
            f"viewgraph and were skipped in the reconstruction:\n"
            f"{epo.images_not_in_viewgraph}"
        )
    return reconstruction
