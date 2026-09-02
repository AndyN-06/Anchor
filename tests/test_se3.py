"""Tests for rigid transforms, including the frame-convention guards."""

import numpy as np
import pytest

from anchor.geometry.rotations import exp_so3
from anchor.geometry.se3 import SE3, pose_error, relative_pose

RNG = np.random.default_rng(1)


def random_se3(rng, max_angle=np.pi * 0.999):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return SE3(exp_so3(axis * rng.uniform(0, max_angle)), rng.normal(size=3) * 2.0)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0, 3.0, np.pi - 1e-6])
def test_exp_log_roundtrip_across_scales(angle):
    """Regression test for a real bug: the closed form for V^-1 loses all
    precision below ~1e-2 rad because it subtracts two nearly equal numbers and
    then divides by theta^2. At theta = 1e-7 the recovered translation was off
    by 2e-3 m -- 2 mm of pure numerical error, larger than the accuracy this
    whole project is trying to achieve."""
    for _ in range(30):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        xi = np.concatenate([RNG.normal(size=3) * 2.0, axis * angle])
        assert np.allclose(SE3.exp(xi).log(), xi, atol=1e-9)


def test_inverse_is_a_true_inverse():
    for _ in range(200):
        T = random_se3(RNG)
        assert np.allclose((T @ T.inverse()).matrix(), np.eye(4), atol=1e-12)
        P = RNG.normal(size=(10, 3))
        assert np.allclose(T.inverse().transform(T.transform(P)), P, atol=1e-12)


def test_inverse_rotates_the_translation():
    """(R, t)^-1 is (R^T, -R^T t), not (R^T, -t). The naive version is only
    correct when R = I, so a test with a nonzero rotation is essential."""
    T = SE3(exp_so3(np.array([0.0, 0.0, np.pi / 2])), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(T.inverse().t, np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert not np.allclose(T.inverse().t, -T.t)


def test_composition_is_associative_and_matches_matrix_product():
    A, B, C = random_se3(RNG), random_se3(RNG), random_se3(RNG)
    assert np.allclose(((A @ B) @ C).matrix(), (A @ (B @ C)).matrix(), atol=1e-12)
    assert np.allclose((A @ B).matrix(), A.matrix() @ B.matrix(), atol=1e-12)


def test_transform_and_rotate_differ_for_directions():
    """rotate() must not apply the translation -- a direction has no position."""
    T = random_se3(RNG)
    v = RNG.normal(size=(5, 3))
    assert np.allclose(T.transform(v) - T.rotate(v), T.t)


def test_frame_labels_catch_inverted_transforms():
    T_world_cam = SE3.identity("world", "cam")
    T_cam_imu = SE3.identity("cam", "imu")
    assert (T_world_cam @ T_cam_imu).to_frame == "world"
    assert (T_world_cam @ T_cam_imu).from_frame == "imu"
    with pytest.raises(ValueError, match="frame mismatch"):
        T_world_cam @ T_world_cam           # the classic forgot-to-invert bug
    assert T_world_cam.inverse().to_frame == "cam"


def test_unlabelled_transforms_compose_without_complaint():
    """Labels are optional; requiring them everywhere would be unusable."""
    assert isinstance(random_se3(RNG) @ random_se3(RNG), SE3)


def test_matrix_roundtrip_and_validation():
    T = random_se3(RNG)
    assert np.allclose(SE3.from_matrix(T.matrix()).matrix(), T.matrix())
    bad = T.matrix()
    bad[3] = [0, 0, 1, 1]
    with pytest.raises(ValueError):
        SE3.from_matrix(bad)


def test_adjoint_relates_left_and_right_updates():
    """T @ exp(xi) == exp(Adj(T) xi) @ T -- the identity that lets you convert
    between right- and left-multiplied pose updates in an optimiser."""
    T = random_se3(RNG)
    xi = RNG.normal(size=6) * 0.1
    assert np.allclose((T @ SE3.exp(xi)).matrix(),
                       (SE3.exp(T.adjoint() @ xi) @ T).matrix(), atol=1e-9)


def test_relative_pose_and_error():
    T_world_a, T_world_b = random_se3(RNG), random_se3(RNG)
    T_a_b = relative_pose(T_world_a, T_world_b)
    assert np.allclose((T_world_a @ T_a_b).matrix(), T_world_b.matrix(), atol=1e-12)
    assert pose_error(T_world_a, T_world_a) == pytest.approx((0.0, 0.0), abs=1e-12)
    shifted = SE3(T_world_a.R, T_world_a.t + np.array([0.03, 0.0, 0.0]))
    d_t, d_r = pose_error(shifted, T_world_a)
    assert d_t == pytest.approx(0.03, abs=1e-9)
    assert d_r == pytest.approx(0.0, abs=1e-12)


def test_normalized_repairs_drift_from_long_chains():
    T = random_se3(RNG)
    drifted = SE3(T.R + 1e-6 * RNG.normal(size=(3, 3)), T.t)
    assert not drifted.is_valid(tol=1e-9)
    assert drifted.normalized().is_valid()
