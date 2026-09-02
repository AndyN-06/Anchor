"""Rigid-body transforms in SE(3), with explicit frame naming.

Naming convention (used in every file in this repo)
---------------------------------------------------
``T_A_B`` is the transform that takes a point *expressed in frame B* and
re-expresses it *in frame A*::

    p_A = T_A_B * p_B

Two consequences worth internalising, because reversing either is the single
most common bug in a pose pipeline:

1. Composition chains cancel adjacent frames::

       T_A_C = T_A_B @ T_B_C

2. ``T_world_cam`` (the camera *pose*: where the camera sits in the world) and
   ``T_cam_world`` (the camera *extrinsics*: what you need to project a world
   point) are inverses of each other.  The translation column of ``T_world_cam``
   is the camera centre in world coordinates.  The translation column of
   ``T_cam_world`` is **not** a position -- it is ``-R_cam_world @ C_world``.
   Datasets disagree about which one they publish: TUM RGB-D ground truth is
   ``T_world_cam``, while OpenCV's ``rvec``/``tvec`` from ``solvePnP`` are
   ``T_cam_world``.

Units are metres.  Angles are radians.  Points are (N, 3) rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .rotations import exp_so3, is_rotation, log_so3, project_to_so3, skew

# See rotations.py for why this is 1e-4 and not machine epsilon: the closed
# forms below divide by (1 - cos t), which loses most of its significant digits
# to cancellation long before t reaches zero.
_TAYLOR_THRESH = 1e-2


@dataclass
class SE3:
    """A rigid transform: rotation R (3x3) and translation t (3,).

    Optional ``to_frame`` / ``from_frame`` labels are free-form strings
    (e.g. ``"world"``, ``"cam0"``).  They are not required, but when both
    operands of a composition carry labels the middle frames are checked, which
    catches inverted-transform bugs at the point they happen instead of 200
    lines later as a plausible-looking but wrong pose.
    """

    R: np.ndarray
    t: np.ndarray
    to_frame: Optional[str] = field(default=None)
    from_frame: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    # ------------------------------------------------------------------ ctors
    @classmethod
    def identity(cls, to_frame: Optional[str] = None, from_frame: Optional[str] = None) -> "SE3":
        return cls(np.eye(3), np.zeros(3), to_frame, from_frame)

    @classmethod
    def from_matrix(cls, M: np.ndarray, to_frame=None, from_frame=None) -> "SE3":
        """Build from a 4x4 homogeneous matrix [[R, t], [0, 1]]."""
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got {M.shape}")
        if not np.allclose(M[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"bottom row must be [0,0,0,1], got {M[3]}")
        return cls(M[:3, :3], M[:3, 3], to_frame, from_frame)

    @classmethod
    def from_rotvec(cls, rotvec, t, to_frame=None, from_frame=None) -> "SE3":
        """Build from an axis-angle vector (OpenCV's ``rvec``) and a translation."""
        return cls(exp_so3(rotvec), t, to_frame, from_frame)

    # ------------------------------------------------------------- accessors
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous form."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def rotvec(self) -> np.ndarray:
        """Rotation as an axis-angle vector (OpenCV's ``rvec``)."""
        return log_so3(self.R)

    @property
    def translation(self) -> np.ndarray:
        return self.t

    def is_valid(self, tol: float = 1e-6) -> bool:
        return is_rotation(self.R, tol) and np.all(np.isfinite(self.t))

    def normalized(self) -> "SE3":
        """Re-orthonormalise R after long chains of multiplications."""
        return SE3(project_to_so3(self.R), self.t, self.to_frame, self.from_frame)

    # -------------------------------------------------------------- algebra
    def inverse(self) -> "SE3":
        """Inverse transform, with frames swapped.

        From p_A = R p_B + t, solving for p_B gives p_B = R^T p_A - R^T t.
        Note this is *not* (R^T, -t): forgetting the rotation on the
        translation is the classic sign-of-life bug in pose code.
        """
        Rt = self.R.T
        return SE3(Rt, -Rt @ self.t, to_frame=self.from_frame, from_frame=self.to_frame)

    def __matmul__(self, other: "SE3") -> "SE3":
        """Compose: ``T_A_B @ T_B_C == T_A_C``."""
        if not isinstance(other, SE3):
            return NotImplemented
        if (
            self.from_frame is not None
            and other.to_frame is not None
            and self.from_frame != other.to_frame
        ):
            raise ValueError(
                f"frame mismatch composing T_{self.to_frame}_{self.from_frame} @ "
                f"T_{other.to_frame}_{other.from_frame}: "
                f"'{self.from_frame}' != '{other.to_frame}'"
            )
        return SE3(
            self.R @ other.R,
            self.R @ other.t + self.t,
            to_frame=self.to_frame,
            from_frame=other.from_frame,
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Apply to (N, 3) points: p_A = R p_B + t, written for row vectors."""
        P = np.asarray(points, dtype=float).reshape(-1, 3)
        return P @ self.R.T + self.t

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the rotation only -- for directions, normals, bearing vectors.

        A direction has no position, so translating it is meaningless.  Using
        :meth:`transform` on a ray direction is a real and silent bug.
        """
        V = np.asarray(vectors, dtype=float).reshape(-1, 3)
        return V @ self.R.T

    # ------------------------------------------------------------- Lie group
    @classmethod
    def exp(cls, xi: np.ndarray, to_frame=None, from_frame=None) -> "SE3":
        """Exponential map from the 6-vector xi = [rho (3), omega (3)].

        Translation first, rotation second (the Sophus/Eigen ordering).

            R = exp_so3(omega)
            t = V(omega) @ rho,  V = I + (1-cos t)/t^2 [w]_x + (t - sin t)/t^3 [w]_x^2

        V is *not* the identity: a pure translation in the tangent space traces
        an arc when the rotation is nonzero, and V accounts for that curvature.
        This is the update you apply in an iterative pose refinement -- the
        optimiser solves for a 6-vector increment xi and applies
        ``T <- SE3.exp(xi) @ T``, which keeps the result on the manifold exactly,
        instead of adding numbers to a rotation matrix and hoping it stays
        orthonormal.
        """
        xi = np.asarray(xi, dtype=float).reshape(6)
        rho, omega = xi[:3], xi[3:]
        theta = float(np.linalg.norm(omega))
        K = skew(omega)
        if theta < _TAYLOR_THRESH:
            t2 = theta * theta
            b = 0.5 - t2 / 24.0 + t2 * t2 / 720.0             # (1 - cos t)/t^2
            c = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0     # (t - sin t)/t^3
        else:
            b = (1.0 - np.cos(theta)) / theta**2
            c = (theta - np.sin(theta)) / theta**3
        V = np.eye(3) + b * K + c * (K @ K)
        return cls(exp_so3(omega), V @ rho, to_frame, from_frame)

    def log(self) -> np.ndarray:
        """Inverse of :meth:`exp`: returns xi = [rho (3), omega (3)]."""
        omega = log_so3(self.R)
        theta = float(np.linalg.norm(omega))
        K = skew(omega)
        if theta < _TAYLOR_THRESH:
            # coeff = 1/12 + t^2/720 + t^4/30240 + ...
            # The closed form below is 1 - (something that tends to 1), divided
            # by t^2.  At t = 1e-7 that subtraction destroys every significant
            # digit and 1/t^2 = 1e14 multiplies the garbage up into the result:
            # the recovered translation came out wrong in the third decimal.
            t2 = theta * theta
            coeff = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
        else:
            coeff = (1.0 / theta**2) * (
                1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta)))
            )
        V_inv = np.eye(3) - 0.5 * K + coeff * (K @ K)
        return np.concatenate([V_inv @ self.t, omega])

    def adjoint(self) -> np.ndarray:
        """6x6 adjoint: maps a tangent vector from the right side to the left.

        ``T @ exp(xi) == exp(Adj(T) @ xi) @ T``.  Needed when you switch between
        left- and right-multiplied pose updates, or transport a pose covariance
        between frames.
        """
        A = np.zeros((6, 6))
        A[:3, :3] = self.R
        A[:3, 3:] = skew(self.t) @ self.R
        A[3:, 3:] = self.R
        return A

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        name = "SE3"
        if self.to_frame or self.from_frame:
            name = f"T_{self.to_frame}_{self.from_frame}"
        deg = np.degrees(np.linalg.norm(self.rotvec()))
        return f"{name}(t={np.round(self.t, 4).tolist()} m, angle={deg:.2f} deg)"


def relative_pose(T_world_a: SE3, T_world_b: SE3) -> SE3:
    """``T_a_b``: pose of b expressed in a's frame. This is the RPE building block."""
    return T_world_a.inverse() @ T_world_b


def pose_error(T_est: SE3, T_gt: SE3) -> tuple[float, float]:
    """(translation error in metres, rotation error in radians) between two poses.

    Both inputs must be in the *same* direction convention -- comparing a
    ``T_world_cam`` against a ``T_cam_world`` produces a number, just not a
    meaningful one.
    """
    delta = T_gt.inverse() @ T_est
    return float(np.linalg.norm(delta.t)), float(np.linalg.norm(log_so3(delta.R)))
