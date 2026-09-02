"""Tests for the pinhole + Brown-Conrady camera model, cross-checked against OpenCV."""

import cv2
import numpy as np
import pytest

from anchor.geometry.camera import PinholeCamera
from anchor.geometry.rotations import exp_so3
from anchor.geometry.se3 import SE3

RNG = np.random.default_rng(2)

# Real intrinsics: TUM RGB-D freiburg1 (Kinect v1). Strong radial distortion,
# which is the point -- an ideal pinhole would not exercise the hard paths.
TUM_FR1 = PinholeCamera(517.306408, 516.469215, 318.643040, 255.313989,
                        640, 480, [0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
IDEAL = PinholeCamera.from_fov(60.0, 640, 480)


def random_points_in_front(rng, n=300):
    return rng.uniform([-1.5, -1.5, 0.5], [1.5, 1.5, 6.0], size=(n, 3))


@pytest.mark.parametrize("cam", [IDEAL, TUM_FR1], ids=["ideal", "tum_fr1"])
def test_projection_matches_opencv(cam):
    """The reference implementation for projection is cv2.projectPoints. If ours
    disagrees, ours is wrong."""
    P_world = random_points_in_front(RNG)
    T_cam_world = SE3(exp_so3(np.array([0.1, -0.2, 0.3])), np.array([0.05, -0.1, 0.2]))

    uv = cam.project_world(P_world, T_cam_world, return_mask=False)
    uv_cv, _ = cv2.projectPoints(P_world, T_cam_world.rotvec(), T_cam_world.t,
                                 cam.K, cam.dist)
    assert np.allclose(uv, uv_cv.reshape(-1, 2), atol=1e-9)


def test_rvec_tvec_convention_is_T_cam_world():
    """OpenCV's (rvec, tvec) is the extrinsic T_cam_world, not the camera pose.
    This test is the executable statement of that fact -- passing T_world_cam
    below gives a plausible but wrong image."""
    P_world = random_points_in_front(RNG, 50)
    T_world_cam = SE3(exp_so3(np.array([0.2, 0.4, -0.1])), np.array([1.0, 0.5, -0.3]),
                      "world", "cam")
    T_cam_world = T_world_cam.inverse()

    uv_right = IDEAL.project_world(P_world, T_cam_world, return_mask=False)
    uv_cv, _ = cv2.projectPoints(P_world, T_cam_world.rotvec(), T_cam_world.t,
                                 IDEAL.K, IDEAL.dist)
    assert np.allclose(uv_right, uv_cv.reshape(-1, 2), atol=1e-9)

    uv_wrong = IDEAL.project_world(P_world, T_world_cam, return_mask=False)
    assert not np.allclose(uv_right, uv_wrong, atol=1.0)


def test_camera_centre_projects_nowhere_but_the_pose_translation_is_the_centre():
    T_world_cam = SE3(exp_so3(np.array([0.3, 0.1, 0.7])), np.array([2.0, -1.0, 1.5]),
                      "world", "cam")
    centre = T_world_cam.t
    # The camera centre maps to the origin of the camera frame: z = 0, not imageable.
    p_cam = T_world_cam.inverse().transform(centre)
    assert np.allclose(p_cam, 0.0, atol=1e-12)
    _, valid = IDEAL.project(p_cam)
    assert not valid[0]
    # And the extrinsic translation is NOT the camera position.
    assert not np.allclose(T_world_cam.inverse().t, centre)


@pytest.mark.parametrize("cam", [IDEAL, TUM_FR1], ids=["ideal", "tum_fr1"])
def test_pixel_to_normalized_inverts_projection(cam):
    """Round trip over the whole image, including the corners where distortion
    is strongest."""
    uv = np.stack([RNG.uniform(0, cam.width - 1, 2000),
                   RNG.uniform(0, cam.height - 1, 2000)], axis=1)
    xy = cam.pixel_to_normalized(uv)
    rays = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    assert np.allclose(cam.project(rays, return_mask=False), uv, atol=1e-9)


def test_undistortion_converges_everywhere_in_the_image():
    uv = np.stack([RNG.uniform(0, 639, 3000), RNG.uniform(0, 479, 3000)], axis=1)
    xy_d = np.stack([(uv[:, 0] - TUM_FR1.cx) / TUM_FR1.fx,
                     (uv[:, 1] - TUM_FR1.cy) / TUM_FR1.fy], axis=1)
    _, converged = TUM_FR1.undistort_normalized(xy_d, return_converged=True)
    assert converged.all()


def test_undistortion_agrees_with_opencv_inside_the_image():
    uv = np.stack([RNG.uniform(0, 639, 500), RNG.uniform(0, 479, 500)], axis=1)
    mine = TUM_FR1.pixel_to_normalized(uv)
    theirs = cv2.undistortPoints(uv.reshape(-1, 1, 2), TUM_FR1.K, TUM_FR1.dist).reshape(-1, 2)
    # OpenCV's default is 5 fixed-point iterations, so it is the looser of the
    # two; agreement to 1e-4 normalised (~0.05 px) is as much as it can offer.
    assert np.abs(mine - theirs).max() < 1e-4


def test_distortion_jacobian_matches_finite_differences():
    xy = RNG.uniform(-0.7, 0.7, size=(200, 2))
    J = TUM_FR1._distortion_jacobian(xy)
    h = 1e-6
    for axis in (0, 1):
        step = np.zeros((1, 2))
        step[0, axis] = h
        num = (TUM_FR1.distort_normalized(xy + step)
               - TUM_FR1.distort_normalized(xy - step)) / (2 * h)
        assert np.allclose(J[:, :, axis], num, atol=1e-6)


def test_points_behind_camera_are_marked_invalid():
    """A point at -z projects to a perfectly reasonable-looking pixel. The mask
    is the only thing standing between that and a phantom correspondence."""
    P = np.array([[0.3, 0.2, -2.0], [0.3, 0.2, 2.0], [0.0, 0.0, 0.0]])
    uv, valid = IDEAL.project(P)
    assert valid.tolist() == [False, True, False]
    assert np.all(np.isfinite(uv[0]))  # it did produce a number, that's the trap


def test_points_outside_the_sensor_are_marked_invalid():
    P = np.array([[10.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    _, valid = IDEAL.project(P)
    assert valid.tolist() == [False, True]


def test_principal_point_projects_to_itself():
    uv, valid = IDEAL.project(np.array([[0.0, 0.0, 3.7]]))
    assert valid[0]
    assert np.allclose(uv[0], [IDEAL.cx, IDEAL.cy], atol=1e-12)


def test_depth_does_not_change_the_pixel_along_a_ray():
    """Perspective projection is scale-invariant along a viewing ray: this is
    exactly why a single image cannot recover depth."""
    p = np.array([0.4, -0.2, 1.0])
    uv = IDEAL.project(np.stack([p * s for s in (1.0, 2.5, 17.0)]), return_mask=False)
    assert np.allclose(uv - uv[0], 0.0, atol=1e-12)


def test_scaled_intrinsics_are_consistent_with_resizing():
    s = 0.5
    small = TUM_FR1.scaled(s)
    P = random_points_in_front(RNG, 200)
    T = SE3.identity()
    uv_full = TUM_FR1.project_world(P, T, return_mask=False)
    uv_small = small.project_world(P, T, return_mask=False)
    # Half-pixel-centre convention: u_small = (u_full + 0.5) * s - 0.5
    assert np.allclose(uv_small, (uv_full + 0.5) * s - 0.5, atol=1e-9)
    assert np.allclose(small.dist, TUM_FR1.dist)  # distortion is resolution-free


def test_fov_and_focal_length_are_consistent():
    cam = PinholeCamera.from_fov(90.0, 800, 600)
    assert cam.fx == pytest.approx(400.0)          # tan(45 deg) = 1
    assert cam.fov_deg()[0] == pytest.approx(90.0)


def test_reprojection_error_is_zero_on_exact_data():
    P = random_points_in_front(RNG, 100)
    T_cam_world = SE3(exp_so3(np.array([0.05, 0.1, -0.2])), np.array([0.1, 0.0, 0.3]))
    uv = TUM_FR1.project_world(P, T_cam_world, return_mask=False)
    assert TUM_FR1.reprojection_error(P, uv, T_cam_world).max() < 1e-9
    # A 1 cm sideways shift at ~3 m with f=517 px should move points by ~1.7 px.
    shifted = SE3(T_cam_world.R, T_cam_world.t + np.array([0.01, 0, 0]))
    assert TUM_FR1.reprojection_error(P, uv, shifted).mean() > 0.5


def test_json_roundtrip(tmp_path):
    path = tmp_path / "intrinsics.json"
    TUM_FR1.to_json(path)
    loaded = PinholeCamera.from_json(path)
    assert np.allclose(loaded.K, TUM_FR1.K)
    assert np.allclose(loaded.dist, TUM_FR1.dist)
    assert (loaded.width, loaded.height) == (TUM_FR1.width, TUM_FR1.height)
