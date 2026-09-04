"""Umeyama alignment: exactness, the reflection trap, and degeneracy."""

import numpy as np
import pytest

from anchor.eval.align import alignment_is_degenerate, umeyama_alignment
from anchor.geometry.rotations import exp_so3

RNG = np.random.default_rng(21)


def test_recovers_a_known_rigid_transform_exactly():
    X = RNG.normal(size=(100, 3)) * 3.0
    R, t = exp_so3(np.array([0.4, -0.7, 1.2])), np.array([2.0, -5.0, 0.5])
    Y = X @ R.T + t
    T, s = umeyama_alignment(X, Y, with_scale=False)
    assert s == 1.0
    assert np.allclose(T.R, R, atol=1e-12)
    assert np.allclose(T.t, t, atol=1e-12)


def test_recovers_a_known_similarity_transform_exactly():
    X = RNG.normal(size=(100, 3)) * 3.0
    R, t, s_true = exp_so3(np.array([-0.2, 0.9, 0.3])), np.array([1.0, 1.0, 1.0]), 0.37
    Y = s_true * (X @ R.T) + t
    T, s = umeyama_alignment(X, Y, with_scale=True)
    assert s == pytest.approx(s_true, rel=1e-12)
    assert np.allclose(T.R, R, atol=1e-12)
    assert np.allclose(T.t, t, atol=1e-12)


def test_se3_alignment_leaves_scale_error_visible():
    """The whole point of choosing SE(3) for a metric system: a system that
    reports every distance 40% too short must NOT be rescaled into looking
    correct. Sim(3) would absorb it; SE(3) must not."""
    X = RNG.normal(size=(100, 3)) * 3.0
    Y = 0.6 * X
    T_se3, _ = umeyama_alignment(X, Y, with_scale=False)
    resid_se3 = np.linalg.norm((X @ T_se3.R.T) + T_se3.t - Y, axis=1)
    T_sim3, s = umeyama_alignment(X, Y, with_scale=True)
    resid_sim3 = np.linalg.norm(s * (X @ T_sim3.R.T) + T_sim3.t - Y, axis=1)
    assert resid_se3.max() > 0.5
    assert resid_sim3.max() < 1e-12
    assert s == pytest.approx(0.6)


def test_rejects_reflections():
    """A mirrored point set has no proper rotation that fits it. The naive
    U @ V.T would return det = -1 and a suspiciously good residual, fitting the
    mirror image. Umeyama's S correction must force a genuine rotation."""
    X = RNG.normal(size=(80, 3))
    Y = X * np.array([1.0, 1.0, -1.0])          # reflection through the xy plane
    T, _ = umeyama_alignment(X, Y)
    assert np.isclose(np.linalg.det(T.R), 1.0, atol=1e-9)
    # And it should NOT fit well, because no rotation can produce a reflection.
    assert np.linalg.norm((X @ T.R.T) + T.t - Y, axis=1).max() > 0.1


def test_alignment_is_least_squares_optimal_under_noise():
    """With noise, the recovered transform should beat any nearby perturbation."""
    X = RNG.normal(size=(200, 3)) * 2.0
    R, t = exp_so3(np.array([0.1, 0.2, 0.3])), np.array([0.5, 0.0, -0.5])
    Y = X @ R.T + t + RNG.normal(scale=0.01, size=(200, 3))
    T, _ = umeyama_alignment(X, Y)

    def cost(Rm, tm):
        return float((((X @ Rm.T) + tm - Y) ** 2).sum())

    best = cost(T.R, T.t)
    for _ in range(50):
        dR = exp_so3(RNG.normal(scale=0.01, size=3))
        dt = RNG.normal(scale=0.01, size=3)
        assert cost(T.R @ dR, T.t + dt) >= best - 1e-12


def test_requires_enough_points_and_matching_shapes():
    with pytest.raises(ValueError, match="at least 3"):
        umeyama_alignment(RNG.normal(size=(2, 3)), RNG.normal(size=(2, 3)))
    with pytest.raises(ValueError, match="shape mismatch"):
        umeyama_alignment(RNG.normal(size=(5, 3)), RNG.normal(size=(6, 3)))


def test_detects_degenerate_geometry():
    """A straight-line trajectory leaves rotation about that line unconstrained.
    The SVD still returns an answer; it is just arbitrary in one DoF."""
    line = np.linspace(0, 1, 30)[:, None] * np.array([1.0, 2.0, -0.5])
    assert alignment_is_degenerate(line)
    assert alignment_is_degenerate(np.zeros((10, 3)))
    assert not alignment_is_degenerate(RNG.normal(size=(30, 3)))


def test_coincident_points_cannot_define_a_scale():
    X = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    with pytest.raises(ValueError, match="coincident"):
        umeyama_alignment(X, X, with_scale=True)
