"""Synthetic scenes with exact ground truth.

Why this exists before any real data: PnP, RANSAC and pose refinement all have
the property that a broken implementation still returns a plausible-looking
pose.  On real images you cannot tell a 3 cm bias from a convention error from
bad correspondences.  Here the true pose is known exactly, noise is dialled in
one source at a time, and a solver that does not recover the answer to ~1e-9
with zero noise is simply wrong.

Everything is metres and radians.  Poses returned as ``T_world_cam`` (camera
pose in the world); call ``.inverse()`` to get the ``T_cam_world`` extrinsics
that projection wants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera
from .se3 import SE3

WORLD_UP = np.array([0.0, 0.0, 1.0])  # z-up world, as in most SfM/CAD conventions


def look_at(eye, target, world_up=WORLD_UP) -> SE3:
    """Camera pose ``T_world_cam`` for a camera at ``eye`` looking at ``target``.

    Builds the rotation from its columns, using the fact that the columns of
    ``R_world_cam`` are the camera's own axes expressed in world coordinates:

        column 2 (z) = forward  = normalize(target - eye)
        column 1 (y) = down     = the part of -world_up perpendicular to forward
        column 0 (x) = right    = down x forward

    The cross product is ordered that way because the camera frame is
    x-right / y-DOWN / z-forward (OpenCV), so x = y * z for a right-handed
    frame.  Using world_up directly as the y column would give a left-handed
    frame with det(R) = -1, which then mirrors every subsequent projection.
    """
    eye = np.asarray(eye, float).reshape(3)
    target = np.asarray(target, float).reshape(3)
    world_up = np.asarray(world_up, float).reshape(3)

    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-12:
        raise ValueError("look_at: eye and target coincide, view direction undefined")
    forward /= n

    down = -world_up - np.dot(-world_up, forward) * forward   # Gram-Schmidt
    n = np.linalg.norm(down)
    if n < 1e-9:
        raise ValueError("look_at: view direction is parallel to world_up (gimbal degenerate)")
    down /= n

    right = np.cross(down, forward)
    R_world_cam = np.column_stack([right, down, forward])
    return SE3(R_world_cam, eye, to_frame="world", from_frame="cam")


def random_point_cloud(n: int, lo=(-2.0, -2.0, 0.0), hi=(2.0, 2.0, 2.5), rng=None) -> np.ndarray:
    """(n, 3) uniformly random world points inside an axis-aligned box."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(np.asarray(lo, float), np.asarray(hi, float), size=(n, 3))


def orbit_poses(n: int, radius: float = 3.0, height: float = 1.5,
                target=(0.0, 0.0, 1.0), start_deg: float = 0.0,
                sweep_deg: float = 360.0, world_up=WORLD_UP) -> list[SE3]:
    """``n`` camera poses on a circle around ``target``, all looking inward.

    A reasonable stand-in for walking around a room with a phone.  Note this
    trajectory has good *parallax* -- viewpoints genuinely differ -- which is
    what triangulation needs.  A pure rotation (camera spinning in place) gives
    zero parallax and no depth information at all; that degenerate case is
    worth testing separately, and is a real failure mode when someone
    relocalizes by turning on the spot.
    """
    target = np.asarray(target, float)
    angles = np.radians(start_deg + np.linspace(0.0, sweep_deg, n, endpoint=(sweep_deg % 360 != 0)))
    poses = []
    for a in angles:
        eye = target + np.array([radius * np.cos(a), radius * np.sin(a), 0.0])
        eye[2] = height
        poses.append(look_at(eye, target, world_up))
    return poses


@dataclass
class Observations:
    """2D measurements from one camera, with ground truth about each one.

    ``uv[i]`` is the pixel where ``point_ids[i]`` was seen.  ``is_outlier[i]``
    marks measurements that were deliberately corrupted -- so a RANSAC
    implementation can be scored on whether it actually found the right
    inlier set, not just on whether the final pose looks acceptable.
    """

    uv: np.ndarray            # (M, 2) pixels
    point_ids: np.ndarray     # (M,) index into the world point array
    is_outlier: np.ndarray    # (M,) bool


def observe(camera: PinholeCamera, T_cam_world: SE3, points_world: np.ndarray,
            noise_px: float = 0.0, outlier_frac: float = 0.0,
            rng=None) -> Observations:
    """Project world points into one camera and corrupt the result realistically.

    Three things happen, in the order they happen physically:

    1. Visibility. Points behind the camera or outside the sensor are dropped.
       This is not cosmetic: a projection routine that keeps points with z < 0
       will happily hand a solver correspondences from behind the camera.
    2. Measurement noise. Gaussian, isotropic, in *pixels* -- because the noise
       is introduced by the detector on the image grid, not in the world.  A
       sub-pixel keypoint detector is typically 0.5-1.5 px here.
    3. Outliers. A fraction of measurements get replaced by a uniformly random
       pixel, standing in for a wrong feature match.  Real mismatch outliers
       are not uniform (repeated structure clusters them), so treat this as an
       optimistic case.
    """
    rng = np.random.default_rng() if rng is None else rng
    points_world = np.asarray(points_world, float).reshape(-1, 3)

    uv, valid = camera.project_world(points_world, T_cam_world)
    ids = np.flatnonzero(valid)
    uv = uv[valid]

    if noise_px > 0:
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)

    is_outlier = np.zeros(len(ids), dtype=bool)
    n_out = int(round(outlier_frac * len(ids)))
    if n_out > 0:
        pick = rng.choice(len(ids), size=n_out, replace=False)
        is_outlier[pick] = True
        uv[pick] = np.stack([
            rng.uniform(0, camera.width - 1, n_out),
            rng.uniform(0, camera.height - 1, n_out),
        ], axis=1)

    return Observations(uv=uv, point_ids=ids, is_outlier=is_outlier)


@dataclass
class SyntheticScene:
    """A full ground-truth scene: points, camera, poses, and per-view measurements."""

    points_world: np.ndarray          # (N, 3)
    camera: PinholeCamera
    poses_world_cam: list[SE3]        # camera poses (invert for extrinsics)
    observations: list[Observations]  # one per pose

    def visible_counts(self) -> np.ndarray:
        return np.array([len(o.point_ids) for o in self.observations])


def make_scene(n_points: int = 300, n_views: int = 12, noise_px: float = 0.0,
               outlier_frac: float = 0.0, camera: PinholeCamera | None = None,
               rng=None, **orbit_kwargs) -> SyntheticScene:
    """Convenience builder: a room-sized point cloud viewed from an orbit."""
    rng = np.random.default_rng() if rng is None else rng
    camera = PinholeCamera.from_fov(60.0, 640, 480) if camera is None else camera
    pts = random_point_cloud(n_points, rng=rng)
    poses = orbit_poses(n_views, **orbit_kwargs)
    obs = [observe(camera, T.inverse(), pts, noise_px, outlier_frac, rng) for T in poses]
    return SyntheticScene(pts, camera, poses, obs)
