# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NumPy-array → pycolmap reconstruction helpers (pycolmap 4.x compatible).

Vendored from the VGGT fork's ``vggt/dependency/np_to_pycolmap.py`` so the
original facebookresearch/vggt submodule can stay unmodified. Both conversion
functions target the modern pycolmap rig API (``add_camera_with_trivial_rig``
/ ``add_image_with_trivial_frame``):

- ``batch_np_matrix_to_pycolmap_wo_track`` (the default EPO / non-BA path) is
  copied verbatim from the fork — same exact results.
- ``batch_np_matrix_to_pycolmap`` (BA path) is the fork's function with the
  same three rig-API fixes applied; the fork's version predates the rig API
  and raises ``AttributeError: 'Image' ... cam_from_world is read-only`` on
  pycolmap 4.x (it is never reached by EPO, which runs ``use_ba=False``).

Only ``project_3D_points_np`` is imported lazily from the pristine vggt package
(it lives in ``vggt.dependency.projection``) so this module has no import-order
coupling and stays importable without a configured sys.path.
"""

import numpy as np
import pycolmap


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
    image_names=None,
):
    """Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    3. Supports grouping shared cameras by folder name if image_names is provided.
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        from vggt.dependency.projection import project_3D_points_np

        projected_points_2d, projected_points_cam = project_3D_points_np(
            points3d, extrinsics, intrinsics
        )
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    ## Let it run with whatever is found, do not skip BA any more
    # if masks.sum(1).min() < min_inlier_per_frame:
    #     print(f"Not enough inliers per frame, skip BA.")
    #     return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Track the mapping from valid_idx to point3D_id
    vidx_to_point3D_id = {}

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        point3D_id = reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)
        vidx_to_point3D_id[vidx] = point3D_id

    # Pre-compute cameras if shared_camera is True
    # Maps frame index `fidx` to `camera_id`
    frame_to_camera_id = {}

    if shared_camera:
        if image_names is not None:
            # Group frames by their parent folder (camera_name)
            # Format: camera_name/image_name
            print("Grouping frames by camera name")
            camera_groups = {}
            for idx, name in enumerate(image_names):
                # Extract camera name (folder name)
                camera_name = name.split("/")[0] if "/" in name else "default"
                if camera_name not in camera_groups:
                    camera_groups[camera_name] = []
                camera_groups[camera_name].append(idx)

            # Create one camera per group by averaging intrinsics
            for i, (_cam_name, indices) in enumerate(camera_groups.items()):
                # Collect params for all frames in this group
                params_list = []
                for idx in indices:
                    params_list.append(
                        _build_pycolmap_intri(
                            idx, intrinsics, camera_type, extra_params
                        )
                    )

                # Average parameters
                avg_params = np.mean(params_list, axis=0)

                cam_id = i + 1
                camera = pycolmap.Camera(
                    camera_id=cam_id,
                    model=camera_type,
                    width=int(image_size[0]),
                    height=int(image_size[1]),
                    params=avg_params,
                )
                reconstruction.add_camera_with_trivial_rig(camera)

                # Map frames to this camera
                for idx in indices:
                    frame_to_camera_id[idx] = cam_id
        else:
            # Fallback: Average all frames into a single global camera
            params_list = []
            for idx in range(N):
                params_list.append(
                    _build_pycolmap_intri(idx, intrinsics, camera_type, extra_params)
                )
            avg_params = np.mean(params_list, axis=0)

            cam_id = 1
            camera = pycolmap.Camera(
                camera_id=cam_id,
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=avg_params,
            )
            reconstruction.add_camera_with_trivial_rig(camera)

            for idx in range(N):
                frame_to_camera_id[idx] = cam_id

    # Iterate over frames to add images and cameras (if not shared/pre-computed)
    for fidx in range(N):
        if shared_camera:
            cam_id = frame_to_camera_id[fidx]
        else:
            # Create a unique camera for this frame
            pycolmap_intri = _build_pycolmap_intri(
                fidx, intrinsics, camera_type, extra_params=None
            )
            cam_id = fidx + 1
            camera = pycolmap.Camera(
                camera_id=cam_id,
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
            )
            reconstruction.add_camera_with_trivial_rig(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image_name = (
            image_names[fidx] if image_names is not None else f"image_{fidx + 1}"
        )

        # Create image; the pose is carried by the trivial frame on add (below).
        image = pycolmap.Image(
            image_id=fidx + 1,
            name=image_name,
            camera_id=cam_id,
        )

        points2D_list = []
        point2D_idx = 0

        # Iterate through valid 3D points and check if they're visible in this frame
        for vidx in valid_idx:
            if masks[fidx, vidx]:  # Check if this point is visible in this frame
                point3D_id = vidx_to_point3D_id[vidx]  # Use the actual assigned ID
                point2D_xy = tracks[fidx, vidx]
                points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                # add element
                track = reconstruction.points3D[point3D_id].track
                track.add_element(fidx + 1, point2D_idx)
                point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        # Modern pycolmap rig API: native-list points2D + trivial frame carries
        # the pose (same pattern as batch_np_matrix_to_pycolmap_wo_track).
        image.points2D = points2D_list
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction, valid_mask  # Return valid_mask as well


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.
    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.
    Do NOT use this for BA.
    """
    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    # Add all 3D points first
    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None

    for fidx in range(N):
        # Set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )
            # FIX 1: Modern pycolmap requires registering a trivial rig for the camera
            reconstruction.add_camera_with_trivial_rig(camera)

        # Set camera pose transformation matrix
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )

        # Create image container with structural properties
        image = pycolmap.Image(
            name=f"image_{fidx + 1}",
            image_id=fidx + 1,
            camera_id=camera.camera_id,
        )

        # Build points2D list - just reference the 3D points
        points2D_list = []
        for vidx in range(P):
            frame_idx = int(points_xyf[vidx, 2])
            if frame_idx == fidx:
                xy = points_xyf[vidx, :2]
                point3d_id = vidx + 1  # 1-indexed
                points2D_list.append(pycolmap.Point2D(xy, point3d_id))

        # FIX 2: PyCOLMAP now implicitly handles native Python lists
        image.points2D = points2D_list

        # FIX 3: Hooks the image data, pose, and frame sequence seamlessly into the rig model
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [
                intrinsics[fidx][0, 0],
                intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [
                focal,
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
                extra_params[fidx][0],
            ]
        )
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
