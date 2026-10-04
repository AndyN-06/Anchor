"""Tests for the synthetic scene generator, plus one end-to-end convention check."""

import cv2
import numpy as np
import pytest

from anchor.geometry.camera import PinholeCamera
from anchor.geometry.rotations import is_rotation
from anchor.geometry.se3 import SE3, pose_error
from anchor.geometry.synthetic import (
    look_at, make_scene, observe, orbit_poses, random_point_cloud,
)

RNG = np.random.default_rng(4)
CAM = PinholeCamera.from_fov(60.0, 640, 480)


def test_look_at_is_right_handed_and_points_at_the_target():
    eye, target = np.array([3.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.0])
    T_world_cam = look_at(eye, target)
    assert is_rotation(T_world_cam.R)
    assert np.isclose(np.linalg.det(T_world_cam.R), 1.0)   # not a mirrored frame
    assert np.allclose(T_world_cam.t, eye)                  # translation IS the centre
    forward = T_world_cam.R[:, 2]                           # camera +z in world
    assert np.allclose(forward, (target - eye) / np.linalg.norm(target - eye))
    # The target must land on the principal point.
    uv = CAM.project_world(target[None, :], T_world_cam.inverse(), return_mask=False)
    assert np.allclose(uv[0], [CAM.cx, CAM.cy], atol=1e-9)


def test_look_at_camera_y_axis_points_down_in_the_world():
    """OpenCV camera convention is y-down. With a z-up world, the camera's y
    axis must have a negative world-z component -- if it does not, the image is
    upside down and every subsequent overlay is mirrored."""
    T = look_at(np.array([3.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.5]))
    assert T.R[2, 1] < 0.0


def test_look_at_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        look_at([0, 0, 0], [0, 0, 0])                       # no view direction
    with pytest.raises(ValueError):
        look_at([0, 0, 2], [0, 0, 0])                       # looking straight down z-up


def test_orbit_poses_lie_on_the_circle_and_face_inward():
    target = np.array([0.0, 0.0, 1.0])
    poses = orbit_poses(8, radius=2.5, height=1.2, target=target)
    for T in poses:
        assert np.isclose(np.linalg.norm(T.t[:2] - target[:2]), 2.5)
        assert np.isclose(T.t[2], 1.2)
        to_target = target - T.t
        assert np.dot(T.R[:, 2], to_target) > 0.0          # +z points toward target


def test_noise_free_observations_reproject_exactly():
    scene = make_scene(200, 6, noise_px=0.0, rng=np.random.default_rng(0))
    for T_world_cam, obs in zip(scene.poses_world_cam, scene.observations):
        err = scene.camera.reprojection_error(
            scene.points_world[obs.point_ids], obs.uv, T_world_cam.inverse())
        assert err.max() < 1e-9


def test_all_reported_observations_are_in_front_and_in_bounds():
    scene = make_scene(300, 5, rng=np.random.default_rng(1))
    for T_world_cam, obs in zip(scene.poses_world_cam, scene.observations):
        P_cam = T_world_cam.inverse().transform(scene.points_world[obs.point_ids])
        assert (P_cam[:, 2] > 0).all()
        assert (obs.uv[:, 0] >= -0.5).all() and (obs.uv[:, 0] <= scene.camera.width - 0.5).all()
        assert (obs.uv[:, 1] >= -0.5).all() and (obs.uv[:, 1] <= scene.camera.height - 0.5).all()


def test_noise_has_the_requested_standard_deviation():
    pts = random_point_cloud(4000, rng=np.random.default_rng(2))
    T = look_at([3, 0, 1.5], [0, 0, 1])
    rng = np.random.default_rng(3)
    clean = observe(CAM, T.inverse(), pts, noise_px=0.0, rng=np.random.default_rng(3))
    noisy = observe(CAM, T.inverse(), pts, noise_px=1.0, rng=rng)
    assert np.array_equal(clean.point_ids, noisy.point_ids)
    residual = (noisy.uv - clean.uv).ravel()
    assert residual.std() == pytest.approx(1.0, rel=0.06)


def test_outlier_fraction_and_labelling():
    pts = random_point_cloud(1000, rng=np.random.default_rng(5))
    T = look_at([3, 0, 1.5], [0, 0, 1])
    obs = observe(CAM, T.inverse(), pts, noise_px=0.5, outlier_frac=0.3,
                  rng=np.random.default_rng(6))
    assert obs.is_outlier.mean() == pytest.approx(0.3, abs=0.01)
    # Outliers must actually be far off; inliers must be within a few sigma.
    err = CAM.reprojection_error(pts[obs.point_ids], obs.uv, T.inverse())
    assert err[~obs.is_outlier].max() < 5.0
    assert np.median(err[obs.is_outlier]) > 20.0


def test_opencv_pnp_recovers_the_ground_truth_pose():
    """End-to-end convention check, and the baseline the hand-written PnP will
    be measured against later. If our frames, intrinsics and distortion handling
    are all consistent, an off-the-shelf solver must return the pose we started
    from to solver precision."""
    pts = random_point_cloud(400, rng=np.random.default_rng(7))
    cam = PinholeCamera(517.3, 516.5, 318.6, 255.3, 640, 480,
                        [0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
    T_world_cam = look_at([2.5, 1.0, 1.6], [0, 0, 1.0])
    T_cam_world = T_world_cam.inverse()

    obs = observe(cam, T_cam_world, pts, noise_px=0.0, rng=np.random.default_rng(8))
    ok, rvec, tvec = cv2.solvePnP(pts[obs.point_ids], obs.uv, cam.K, cam.dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    T_est = SE3.from_rotvec(rvec.reshape(3), tvec.reshape(3))
    d_t, d_r = pose_error(T_est, T_cam_world, convention="T_cam_world")
    assert d_t < 1e-6, f"translation off by {d_t*1000:.3f} mm"
    assert np.degrees(d_r) < 1e-4, f"rotation off by {np.degrees(d_r):.5f} deg"


def test_pnp_degrades_gracefully_with_noise_not_catastrophically():
    """Sanity floor for later comparison: with 1 px noise and no outliers, a
    correct pipeline should still be within a couple of centimetres."""
    pts = random_point_cloud(400, rng=np.random.default_rng(9))
    T_world_cam = look_at([2.5, 1.0, 1.6], [0, 0, 1.0])
    obs = observe(CAM, T_world_cam.inverse(), pts, noise_px=1.0,
                  rng=np.random.default_rng(10))
    ok, rvec, tvec = cv2.solvePnP(pts[obs.point_ids], obs.uv, CAM.K, CAM.dist)
    assert ok
    d_t, d_r = pose_error(SE3.from_rotvec(rvec.reshape(3), tvec.reshape(3)),
                          T_world_cam.inverse(), convention="T_cam_world")
    assert d_t < 0.02
    assert np.degrees(d_r) < 0.5
