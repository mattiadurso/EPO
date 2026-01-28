import os
import torch
import pycolmap
import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def get_frustum_strips(scale=0.15, w=1.0, h=1.0):
    # Normalize aspect to fit in scale
    max_dim = max(w, h)
    w_sc = (w / max_dim) * scale * 0.5
    h_sc = (h / max_dim) * scale * 0.5
    z = scale

    # Corners in Camera coords (Right, Down, Forward) -> (+X, +Y, +Z)
    tr = [w_sc, h_sc, z]
    br = [w_sc, -h_sc, z]
    bl = [-w_sc, -h_sc, z]
    tl = [-w_sc, h_sc, z]
    o = [0, 0, 0]

    return [
        [tr, br, bl, tl, tr],  # Image plane
        [o, tr],
        [o, br],
        [o, bl],
        [o, tl],  # Ray to corners
    ]


def log_reconstruction_rerun(
    path,
    entity="",
    static_cameras=False,
    points3D=False,
    static_points=False,
    camera_color=[0, 255, 0],
):
    recon = pycolmap.Reconstruction(path)
    for img_id, img in recon.images.items():
        # COLMAP stores world-to-cam (R, t)
        # Rerun needs cam-to-world for the transform
        R_gt = torch.from_numpy(img.cam_from_world.rotation.matrix())
        t_gt = torch.from_numpy(img.cam_from_world.translation)

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

    dataset = "mipnerf360"
    scene = "bicycle"

    gt_path = f"/home/mattia/Desktop/datasets/{dataset}/{scene}/sparse_150"
    scene_path = f"optimized_reconstruction_GD/{scene}"
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
