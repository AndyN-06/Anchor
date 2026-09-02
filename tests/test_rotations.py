"""Tests for SO(3). Where possible, cross-checked against OpenCV rather than
against my own algebra, so a shared misunderstanding cannot pass silently."""

import cv2
import numpy as np
import pytest

from anchor.geometry.rotations import (
    angle_between_rotations, exp_so3, is_rotation, log_so3, project_to_so3,
    quat_wxyz_to_rot, quat_xyzw_to_rot, rot_to_quat_wxyz, rot_to_quat_xyzw,
    skew, unskew,
)

RNG = np.random.default_rng(0)


def random_rotvec(rng, max_angle=np.pi * 0.999):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0.0, max_angle)


def test_skew_is_cross_product():
    a, b = RNG.normal(size=3), RNG.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))
    assert np.allclose(unskew(skew(a)), a)
    assert np.allclose(skew(a), -skew(a).T)  # skew-symmetric by construction


def test_exp_produces_valid_rotations():
    for _ in range(200):
        R = exp_so3(random_rotvec(RNG))
        assert is_rotation(R)


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 3.0,
                                   np.pi - 1e-6, np.pi - 1e-9])
def test_exp_log_roundtrip_across_scales(angle):
    """The small-angle and near-pi branches are where these formulas break."""
    for _ in range(50):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        omega = axis * angle
        assert np.allclose(log_so3(exp_so3(omega)), omega, atol=1e-9)


def test_log_at_exactly_pi_is_a_valid_inverse():
    """At pi the sign of the axis is genuinely ambiguous; only the matrix must match."""
    for _ in range(50):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = exp_so3(axis * np.pi)
        assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-9)


def test_matches_opencv_rodrigues():
    for _ in range(200):
        omega = random_rotvec(RNG)
        R_cv, _ = cv2.Rodrigues(omega)
        assert np.allclose(exp_so3(omega), R_cv, atol=1e-12)
        rvec_cv, _ = cv2.Rodrigues(R_cv)
        assert np.allclose(log_so3(R_cv), rvec_cv.reshape(3), atol=1e-9)


def test_rotation_composes_and_acts_on_points():
    Ra, Rb = exp_so3(random_rotvec(RNG)), exp_so3(random_rotvec(RNG))
    P = RNG.normal(size=(20, 3))
    # Row-vector form: applying Rb then Ra equals applying (Ra @ Rb).
    assert np.allclose((P @ Rb.T) @ Ra.T, P @ (Ra @ Rb).T)


def test_quaternion_roundtrip_and_ordering():
    for _ in range(200):
        R = exp_so3(random_rotvec(RNG))
        q = rot_to_quat_wxyz(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert q[0] >= 0.0  # canonical sign
        assert np.allclose(quat_wxyz_to_rot(q), R, atol=1e-12)
        # The xyzw helpers must be a pure reordering, not a different rotation.
        q_xyzw = rot_to_quat_xyzw(R)
        assert np.allclose(q_xyzw, np.r_[q[1:], q[0]])
        assert np.allclose(quat_xyzw_to_rot(q_xyzw), R, atol=1e-12)


def test_quaternion_ordering_bug_is_detectable():
    """Feeding a wxyz quaternion to the xyzw reader must give a *different*
    rotation -- otherwise the two orderings are indistinguishable and the TUM
    loader could silently use the wrong one."""
    R = exp_so3(np.array([0.3, -0.7, 1.1]))
    q = rot_to_quat_wxyz(R)
    assert not np.allclose(quat_xyzw_to_rot(q), R, atol=1e-3)


def test_project_to_so3_repairs_drift():
    R = exp_so3(random_rotvec(RNG))
    noisy = R + 1e-3 * RNG.normal(size=(3, 3))
    assert not is_rotation(noisy, tol=1e-6)
    fixed = project_to_so3(noisy)
    assert is_rotation(fixed)
    assert np.linalg.norm(fixed - R) < np.linalg.norm(noisy - R) * 1.5


def test_project_to_so3_rejects_reflections():
    """A reflection has det = -1; the nearest *proper* rotation must not."""
    M = np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(project_to_so3(M)), 1.0)


def test_angle_between_rotations():
    R = exp_so3(random_rotvec(RNG))
    axis = np.array([0.0, 0.0, 1.0])
    delta = exp_so3(axis * 0.25)
    assert np.isclose(angle_between_rotations(R, R @ delta), 0.25)
    assert np.isclose(angle_between_rotations(R, R), 0.0, atol=1e-12)
