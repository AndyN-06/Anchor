"""Trajectory alignment: the Umeyama closed-form solution.

Why alignment is needed at all
------------------------------
A relocalization or SLAM system recovers geometry in *its own* coordinate
frame.  Where that frame's origin sits and how it is oriented is arbitrary --
usually the first camera pose.  Ground truth lives in the motion-capture rig's
frame.  Comparing the two directly measures the difference between two
arbitrary choices of origin, not the quality of the estimate.

So ATE is defined as the error *after* the best rigid alignment between the two
trajectories.  Any residual is error the alignment could not absorb: real drift
and real pose error.

Monocular systems need one more freedom.  A single camera cannot observe
absolute scale -- a room and a dollhouse photographed from proportionally
scaled positions produce identical images -- so a monocular trajectory is
correct only up to a scale factor, and it must be aligned with Sim(3)
(rotation + translation + scale).  RGB-D and stereo *do* observe metric scale,
so they align with SE(3), and estimating a scale factor there would hide a real
error: a system that reports every distance 5% too long is broken, and Sim(3)
alignment would silently rescale that away.  Which alignment you use is
therefore a claim about your system, and it must be stated with the number.
"""

from __future__ import annotations

import numpy as np

from ..geometry.se3 import SE3


def umeyama_alignment(source: np.ndarray, target: np.ndarray,
                      with_scale: bool = False) -> tuple[SE3, float]:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    Solves for (R, t, s) minimising ``sum_i || target_i - (s R source_i + t) ||^2``
    in closed form (Umeyama 1991).  Returns ``(T, s)`` such that::

        target ~= s * (R @ source_i) + t

    Derivation in three steps:

    1. **Centre both clouds.**  For any fixed R and s, the optimal t just lines
       up the centroids, so subtracting the means removes t from the problem.
    2. **Rotation from an SVD.**  What remains is to maximise
       ``trace(R^T Sigma)`` where ``Sigma = (1/n) Y^T X`` is the cross-covariance
       of the centred clouds.  With ``Sigma = U D V^T``, the maximiser is
       ``R = U V^T`` -- this is the orthogonal Procrustes problem.
    3. **Scale from the singular values.**  ``s = trace(D S) / var(source)``.

    The subtlety is step 2's *reflection* case.  ``U V^T`` is orthogonal but not
    necessarily a rotation: when the point sets are related by a mirror (or the
    data is nearly degenerate and noise decides), it comes out with
    ``det = -1``.  Applying that to a trajectory flips handedness, which is
    physically impossible and produces a suspiciously small error -- the
    alignment fits the mirror image better than the truth.  Umeyama's fix is to
    force ``S = diag(1, 1, -1)`` when ``det(U) det(V) < 0``, giving the best
    proper rotation instead.

    Parameters
    ----------
    source, target : (N, 3)
        Corresponding points, same order.  For trajectories these are camera
        positions after timestamp association.
    with_scale : bool
        Sim(3) if True, SE(3) if False.  See the module docstring -- this is a
        claim about whether your system observes metric scale.

    Returns
    -------
    (SE3, float)
        The rigid part and the scale.  Apply with
        ``Trajectory.transformed(T, scale=s)``.
    """
    X = np.asarray(source, float).reshape(-1, 3)
    Y = np.asarray(target, float).reshape(-1, 3)
    if X.shape != Y.shape:
        raise ValueError(f"shape mismatch: source {X.shape} vs target {Y.shape}")
    n = len(X)
    if n < 3:
        raise ValueError(f"need at least 3 correspondences to fix a 3D rotation, got {n}")

    mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mu_x, Y - mu_y

    var_x = float((Xc ** 2).sum() / n)          # mean squared radius of the source cloud
    Sigma = (Yc.T @ Xc) / n                     # (3, 3) cross-covariance

    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0                          # reject the reflection

    R = U @ S @ Vt

    if with_scale:
        if var_x < 1e-15:
            raise ValueError("source points are coincident; scale is undefined")
        s = float(np.trace(np.diag(D) @ S) / var_x)
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    return SE3(R, t, to_frame="target", from_frame="source"), s


def alignment_is_degenerate(points: np.ndarray, tol: float = 1e-6) -> bool:
    """True if the points are collinear (or coincident), so rotation is underdetermined.

    A straight-line trajectory -- a camera pushed down a corridor, or the
    "xyz" style sequences before they turn -- leaves rotation about that line
    unconstrained.  The SVD still returns *an* answer; it is just arbitrary in
    one degree of freedom, and the resulting ATE is not comparable to anything.
    Worth checking before reporting a number.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) < 3:
        return True
    sv = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return bool(sv[1] < tol * max(sv[0], 1e-30))
