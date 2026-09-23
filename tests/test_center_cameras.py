"""``center_cameras_to_identity``: a rigid move that puts one camera at ``[I|0]``.

CPU-only, on a small synthetic ``pycolmap.Reconstruction``.

Run with ``python -m pytest tests/test_center_cameras.py`` or directly.
"""

import os
import sys

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.reconstruction import (  # noqa: E402
    center_cameras_to_identity,
    closest_to_identity,
)


def _recon(n=4, seed=0):
    """``n`` cameras with random poses on one shared camera, plus 5 points."""
    rng = np.random.default_rng(seed)
    rec = pycolmap.Reconstruction()
    rec.add_camera(
        pycolmap.Camera(
            model="PINHOLE", width=100, height=80, params=[60, 60, 50, 40], camera_id=1
        )
    )
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(rec.cameras[1].sensor_id)
    rec.add_rig(rig)
    for i in range(1, n + 1):
        frame = pycolmap.Frame()
        frame.frame_id, frame.rig_id, frame.rig = i, 1, rec.rigs[1]
        R = Rotation.random(random_state=seed + i).as_matrix()
        frame.set_cam_from_world(
            1, pycolmap.Rigid3d(pycolmap.Rotation3d(R), rng.normal(size=3))
        )
        frame.add_data_id(pycolmap.data_t(rec.cameras[1].sensor_id, i))
        rec.add_frame(frame)
        rec.add_image(
            pycolmap.Image(image_id=i, name=f"{i}.jpg", camera_id=1, frame_id=i)
        )
    for _ in range(5):
        rec.add_point3D(rng.normal(size=3), pycolmap.Track(), np.zeros(3, np.uint8))
    return rec


def _poses(rec):
    """``{name: (R, t)}`` world-to-camera."""
    out = {}
    for im in rec.images.values():
        p = im.cam_from_world()
        out[im.name] = (p.rotation.matrix(), p.translation)
    return out


def _in_cameras(rec):
    """Every point expressed in every camera: ``{name: (P, 3)}``."""
    X = np.stack([p.xyz for p in rec.points3D.values()])
    return {n: X @ R.T + t for n, (R, t) in _poses(rec).items()}


def test_anchor_is_exactly_identity():
    rec = _recon()
    assert center_cameras_to_identity(rec, "3.jpg") == "3.jpg"
    R, t = _poses(rec)["3.jpg"]
    assert np.array_equal(R, np.eye(3))
    assert np.array_equal(t, np.zeros(3))


def test_the_move_is_rigid():
    """Relative poses and camera-frame points do not change."""
    rec = _recon()
    before, pts_before = _poses(rec), _in_cameras(rec)
    center_cameras_to_identity(rec, "2.jpg")
    after, pts_after = _poses(rec), _in_cameras(rec)
    for a in before:
        for b in before:
            rel0 = before[a][0] @ before[b][0].T
            rel1 = after[a][0] @ after[b][0].T
            assert np.abs(rel0 - rel1).max() < 1e-12
        assert np.abs(pts_before[a] - pts_after[a]).max() < 1e-12


def test_default_anchor_is_the_camera_closest_to_identity():
    rec = _recon()
    frame = rec.frame(rec.images[4].frame_id)
    near = pycolmap.Rotation3d(Rotation.from_rotvec([0.0, 0.01, 0.0]).as_matrix())
    frame.set_cam_from_world(1, pycolmap.Rigid3d(near, np.array([0.0, 0.0, 0.1])))
    assert center_cameras_to_identity(rec) == "4.jpg"


def test_closest_to_identity_ranks_rotation_then_translation():
    eye = np.eye(3)
    tilted = Rotation.from_rotvec([0.1, 0.0, 0.0]).as_matrix()
    R = np.stack([tilted, eye, eye])
    t = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    assert closest_to_identity(R, t) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
