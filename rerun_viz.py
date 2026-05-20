"""Standalone Rerun visualizer for an EPO run.

Loads up to four COLMAP reconstructions (ground truth, EPO output, BA, BA
refined), aligns them to GT with ``colmap model_aligner`` and logs them as
camera frustums + point clouds in Rerun. Configure the paths at the bottom
of the file before running.
"""

import os

import numpy as np
import pycolmap
import torch

import rerun as rr
import rerun.blueprint as rrb


def get_frustum_strips(
    scale: float = 0.15, w: float = 1.0, h: float = 1.0
) -> list[list[list[float]]]:
    """Return Rerun ``LineStrips3D`` describing a camera frustum.

    Builds a unit pyramid in camera-local coordinates (right-down-forward)
    with apex at the origin and an image plane at depth ``scale``, scaled
    to preserve the ``w:h`` aspect ratio. The first polyline is the image
    plane; the next four are rays from the apex to its corners.
    """
    max_dim = max(w, h)
    w_sc = (w / max_dim) * scale * 0.5
    h_sc = (h / max_dim) * scale * 0.5
    z = scale

    top_right = [w_sc, h_sc, z]
    bot_right = [w_sc, -h_sc, z]
    bot_left = [-w_sc, -h_sc, z]
    top_left = [-w_sc, h_sc, z]
    origin = [0, 0, 0]

    return [
        [top_right, bot_right, bot_left, top_left, top_right],  # image plane
        [origin, top_right],
        [origin, bot_right],
        [origin, bot_left],
        [origin, top_left],
    ]


def log_reconstruction_rerun(
    path: str,
    entity: str = "",
    static_cameras: bool = False,
    points3D: bool = False,
    static_points: bool = False,
    camera_color: list[int] | None = None,
) -> None:
    """Log a COLMAP reconstruction to Rerun under ``world/<entity>``.

    Args:
        path: Path to a COLMAP reconstruction folder.
        entity: Sub-namespace under ``world/`` for these entities.
        static_cameras: If True, log camera transforms as time-static.
        points3D: If True, also log the 3D point cloud.
        static_points: If True, log points as time-static.
        camera_color: RGB color used for the frustum line strips. Defaults to
            green ``[0, 255, 0]`` when ``None``.
    """
    if camera_color is None:
        camera_color = [0, 255, 0]
    recon = pycolmap.Reconstruction(path)
    for _img_id, img in recon.images.items():
        # COLMAP stores world-to-cam (R, t)
        # Rerun needs cam-to-world for the transform
        cfw = img.cam_from_world()
        R_gt = torch.from_numpy(cfw.rotation.matrix())
        t_gt = torch.from_numpy(cfw.translation)

        # C = -R^T * t
        cam_center = -R_gt.T @ t_gt
        cam_rot = R_gt.T  # Rotation from camera to world

        rr.log(
            f"world/{entity}/{img.name}",
            rr.Transform3D(
                translation=cam_center.numpy(),
                mat3x3=cam_rot.numpy(),
            ),
            static=static_cameras,
        )

        # Frustum visualization
        cam = recon.cameras[img.camera_id]
        strips = get_frustum_strips(scale=0.15, w=cam.width, h=cam.height)
        rr.log(
            f"world/{entity}/{img.name}/cam",
            rr.LineStrips3D(strips, colors=camera_color, radii=0.005),
            static=static_cameras,
        )

    # Log GT Point Cloud
    if len(recon.points3D) > 0 and points3D:
        print(f"Logging {len(recon.points3D)} GT points...")
        pts = []
        colors = []
        for p in recon.points3D.values():
            pts.append(p.xyz)
            colors.append(p.color)

        rr.log(
            f"world/{entity}/points",
            rr.Points3D(np.array(pts), colors=np.array(colors), radii=0.01),
            static=static_points,
        )


if __name__ == "__main__":
    dataset = "terrasky3D"
    scene = "vienna_state_opera"

    gt_path = f"/home/mattia/Desktop/datasets/{dataset}/{scene}/sparse_150"
    scene_path = f"optimized_reconstruction/{scene}"
    ba_path = f"benchmarks/vggt_ba/{dataset}/{scene}/sparse"
    ba_ref_path = f"benchmarks/vggt_ba_ref/{dataset}/{scene}/sparse"

    # align all to gt
    for path in [scene_path, ba_path, ba_ref_path]:
        os.system(
            f"colmap model_aligner --input_path {path} --output_path {path} --ref_model_path {gt_path} --alignment_max_error 1"
        )

    rr.init("Feature-Less Optimization", spawn=True)
    # 2. Define the Blueprint: Background color + No Grid
    rr.send_blueprint(
        rrb.Spatial3DView(
            origin="world",
            # Set background to your specific off-white
            background=rrb.Background(kind="SolidColor", color=[251, 251, 255]),
            # This turns off the grid plane visualizer
            line_grid=rrb.LineGrid3D(visible=False),
        ),
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # logs
    log_reconstruction_rerun(gt_path, "gt", camera_color=[28, 186, 81], points3D=True)

    log_reconstruction_rerun(scene_path, "opt", camera_color=[150, 3, 26])
    log_reconstruction_rerun(ba_path, "ba", camera_color=[71, 168, 216])
    log_reconstruction_rerun(ba_ref_path, "ba_ref", camera_color=[9, 73, 110])
