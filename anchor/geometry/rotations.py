"""Rotation utilities for SO(3).

Conventions used everywhere in Anchor
-------------------------------------
* Rotation matrices are 3x3, right-handed, det(R) = +1, R @ R.T = I.
* Points are stored as (N, 3) row vectors (NumPy is row-major, so a point is a
  contiguous row).  A rotation is applied as ``P_rotated = P @ R.T``, which is
  the row-vector form of the textbook column-vector expression ``p' = R p``.
* Angles are radians.  Nothing in this repo takes degrees.
* A rotation vector (axis-angle) ``omega`` is a 3-vector whose direction is the
  rotation axis and whose norm is the rotation angle in radians.  This is the
  same parameterisation OpenCV calls ``rvec``.
"""

from __future__ import annotations

import numpy as np

# Below this angle (radians) the trig coefficients in exp/log are replaced by
# their Taylor expansions.  Two separate reasons, and the second one is the
# subtle one:
#   * at t = 0 exactly, sin(t)/t and (1-cos t)/t^2 are literally 0/0;
#   * for small but nonzero t, (1 - cos t) suffers catastrophic cancellation --
#     at t = 1e-6 the true value is 5e-13 while the two operands are both ~1,
#     so the double-precision result carries ~1e-16 of absolute error, i.e. a
#     relative error of ~2e-4.  Dividing that by t^2 amplifies it further.
# The Taylor series has no such cancellation, and truncating after the t^4 term
# is exact to ~1e-19 at the threshold below, so it is strictly better there.
_TAYLOR_THRESH = 1e-4
_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [v]_x with [v]_x @ u == cross(v, u).

    Geometrically this is the matrix form of the cross product.  It shows up in
    the exponential map because the derivative of a rotation about a fixed axis
    is a cross product with that axis: d/dt (R p) = omega x (R p).
    """
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def unskew(M: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`: pull the 3-vector back out of a skew matrix."""
    M = np.asarray(M, dtype=float)
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map: axis-angle vector -> rotation matrix (Rodrigues formula).

    R = I + sin(t)/t * [w]_x + (1 - cos t)/t^2 * [w]_x^2,  t = |w|

    Why this shape: [w]_x^2 = w w^T - t^2 I projects onto the plane perpendicular
    to the axis.  So the formula says "keep the component along the axis, rotate
    the perpendicular component by t".  The 3 degrees of freedom of the input
    (2 for the axis direction, 1 for the angle) match SO(3)'s 3 DoF exactly,
    which is why this is the natural parameterisation for optimisation: no
    constraints to maintain, unlike the 9 entries of R or the unit-norm
    constraint on a quaternion.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    K = skew(omega)

    if theta < _TAYLOR_THRESH:
        t2 = theta * theta
        A = 1.0 - t2 / 6.0 + t2 * t2 / 120.0        # sin(t)/t
        B = 0.5 - t2 / 24.0 + t2 * t2 / 720.0       # (1 - cos t)/t^2
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)

    return np.eye(3) + A * K + B * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map: rotation matrix -> axis-angle vector. Inverse of :func:`exp_so3`.

    Decompose R into its symmetric and antisymmetric parts:

        (R - R^T) / 2  = sin(t) [a]_x                      (antisymmetric)
        (R + R^T) / 2  = cos(t) I + (1 - cos t) a a^T      (symmetric)

    so ``|unskew(R - R^T)| / 2 = sin t`` and ``(trace(R) - 1) / 2 = cos t``.
    The angle is then ``atan2(sin t, cos t)``.  Using atan2 rather than
    ``arccos(cos t)`` matters: near t = pi, cos t is flat (its derivative
    vanishes), so arccos amplifies floating-point error by ~1/sqrt(eps) ~ 1e8.
    atan2 stays accurate across the whole range.

    Recovering the axis has two bad cases, both where sin(t) -> 0:

    * t ~ 0: the axis is undefined but the *vector* t*a is not, and the
      first-order expansion ``unskew(R - R^T) / 2`` gives it directly.
    * t ~ pi: the antisymmetric part vanishes, so the axis is read off the
      symmetric part instead, where ``a a^T = ((R + R^T)/2 - cos(t) I) / (1 - cos t)``.
      Taking the column of largest norm avoids dividing by a near-zero entry.
      ``a a^T`` is blind to the sign of a, so the sign is restored from the
      antisymmetric part while it still carries signal, and by direct
      reconstruction at exactly pi (where +pi and -pi about an axis are the
      same rotation and the sign is genuinely ambiguous).
    """
    R = np.asarray(R, dtype=float)
    v = unskew(R - R.T)                       # = 2 sin(t) a
    sin_theta = float(np.linalg.norm(v)) / 2.0
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arctan2(sin_theta, cos_theta))

    if theta < _TAYLOR_THRESH:
        # t/(2 sin t) = 1/2 + t^2/12 + 7 t^4/720, no cancellation anywhere.
        t2 = theta * theta
        return (0.5 + t2 / 12.0 + 7.0 * t2 * t2 / 720.0) * v

    if cos_theta > -0.9:                      # comfortably away from pi
        return (theta / (2.0 * sin_theta)) * v

    # Near-pi branch: get the axis from the symmetric part.
    S = ((R + R.T) / 2.0 - cos_theta * np.eye(3)) / (1.0 - cos_theta)  # = a a^T
    idx = int(np.argmax(np.diag(S)))
    axis = S[:, idx] / np.sqrt(max(S[idx, idx], 0.0))
    axis = axis / np.linalg.norm(axis)

    if sin_theta > 1e-12:
        # The antisymmetric part still points along +a; use it to fix the sign.
        if float(np.dot(v, axis)) < 0.0:
            axis = -axis
    else:
        if np.linalg.norm(exp_so3(theta * axis) - R) > np.linalg.norm(
            exp_so3(-theta * axis) - R
        ):
            axis = -axis
    return theta * axis


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """True if R is orthonormal with det = +1 (a proper rotation, not a reflection)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return abs(np.linalg.det(R) - 1.0) < tol


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to M in Frobenius norm, via SVD.

    Needed because repeated multiplication of rotation matrices in floating
    point slowly breaks orthonormality.  M = U S V^T; the closest orthonormal
    matrix is U V^T, and flipping the sign of the last singular vector fixes the
    case where that product comes out as a reflection (det = -1).
    """
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


# --------------------------------------------------------------------------
# Quaternions.
#
# Ordering is a standing source of bugs: this module uses (w, x, y, z) with the
# scalar FIRST, matching Eigen and most SLAM literature.  TUM RGB-D ground-truth
# files and ROS use (x, y, z, w), scalar LAST.  The helpers below are named for
# their ordering so a conversion never happens by accident.
# --------------------------------------------------------------------------


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), with w >= 0.

    Uses Shepperd's method: q and -q represent the same rotation, and the naive
    formula divides by w, which is near zero for rotations near pi.  So pick
    whichever of the four components is largest (from the diagonal of R) and
    divide by that instead, which is always numerically safe.
    """
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:  # canonical sign so comparisons are meaningful
        q = -q
    return q / np.linalg.norm(q)


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) ordering -> rotation matrix. Use for TUM / ROS data."""
    q = np.asarray(q, dtype=float).reshape(4)
    return quat_wxyz_to_rot(np.array([q[3], q[0], q[1], q[2]]))


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (x, y, z, w) ordering. Use when writing TUM-format files."""
    w, x, y, z = rot_to_quat_wxyz(R)
    return np.array([x, y, z, w])


def angle_between_rotations(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle in radians between two rotations.

    This is the rotation part of relative pose error: the angle of the residual
    rotation R_a^T R_b, i.e. how far you must rotate to get from one to the other.
    """
    return float(np.linalg.norm(log_so3(np.asarray(R_a).T @ np.asarray(R_b))))
