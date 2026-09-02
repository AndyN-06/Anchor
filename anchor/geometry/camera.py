"""Pinhole camera model with Brown-Conrady lens distortion.

Camera frame convention (OpenCV, and what this whole repo uses)
--------------------------------------------------------------
    +x right, +y DOWN, +z FORWARD along the optical axis.

Points with z <= 0 are behind the camera and cannot be imaged.  Note this is
*not* the OpenGL / ARCore rendering convention (+y up, -z forward); a virtual
object handed to a renderer needs a 180-degree rotation about x.  Mixing the two
produces an upside-down, backwards overlay -- worth writing down now, because it
will bite at the rendering step.

Pixel convention: pixel (0, 0) is the *centre* of the top-left pixel, matching
OpenCV.  (Some literature puts the origin at the pixel's corner, a half-pixel
offset that quietly biases calibration and reprojection error.)

Projection, in stages
---------------------
    p_cam = (X, Y, Z)                       point in camera frame, metres
    (x, y) = (X/Z, Y/Z)                     perspective divide -> normalised
                                            image plane at Z = 1, unitless
    (x_d, y_d) = distort(x, y)               lens distortion, still unitless
    (u, v) = (fx*x_d + cx, fy*y_d + cy)      pixels

The perspective divide is the only nonlinear step in the ideal model and the
reason a 3D point's image position is not a linear function of its depth --
which is exactly why PnP needs an iterative or algebraic solver rather than a
matrix inverse.

Intrinsics have 4 degrees of freedom here: fx, fy (focal length in pixel units
along each axis -- they differ when pixels are not square), and cx, cy (the
principal point, where the optical axis pierces the sensor, usually near but not
exactly the image centre).  A fifth parameter, skew, is assumed zero; it is
non-zero only for sensors whose pixel axes are not perpendicular, which is
essentially no modern camera.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PinholeCamera:
    """Intrinsics for one camera.

    Parameters
    ----------
    fx, fy, cx, cy : float
        Focal lengths and principal point, in pixels.
    width, height : int
        Image size in pixels, used only for the in-bounds test.
    dist : np.ndarray
        Brown-Conrady coefficients ``(k1, k2, p1, p2, k3)`` in OpenCV's order.
        k* are radial (the barrel/pincushion bulge, symmetric about the
        principal point); p* are tangential (from the lens sitting slightly
        non-parallel to the sensor).  All zeros means an ideal pinhole.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        d = np.asarray(self.dist, dtype=float).reshape(-1)
        if d.size > 5:
            raise ValueError("only the 5-coefficient (k1,k2,p1,p2,k3) model is supported")
        self.dist = np.zeros(5)
        self.dist[: d.size] = d

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_K(cls, K, width, height, dist=None) -> "PinholeCamera":
        K = np.asarray(K, dtype=float)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height,
                   np.zeros(5) if dist is None else dist)

    @classmethod
    def from_fov(cls, fov_x_deg: float, width: int, height: int) -> "PinholeCamera":
        """Ideal camera from a horizontal field of view -- handy for synthetic tests.

        Half the sensor width subtends half the FoV at distance fx, so
        ``fx = (width/2) / tan(fov/2)``.  A wider FoV means a shorter focal
        length, which is why wide-angle lenses distort more.
        """
        fx = (width / 2.0) / np.tan(np.radians(fov_x_deg) / 2.0)
        return cls(fx, fx, (width - 1) / 2.0, (height - 1) / 2.0, width, height)

    # -------------------------------------------------------------- accessors
    @property
    def K(self) -> np.ndarray:
        """3x3 calibration matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist != 0.0))

    def scaled(self, s: float) -> "PinholeCamera":
        """Intrinsics for an image resized by factor s.

        Everything measured in pixels scales; the distortion coefficients do
        not, because they act on *normalised* coordinates, which are
        resolution-independent.  Forgetting to rescale intrinsics after
        downsampling images is a standard source of a constant pose bias.
        """
        return PinholeCamera(
            self.fx * s, self.fy * s,
            # (u + 0.5) maps the pixel *edge*, keeping the half-pixel convention
            # consistent under resampling.
            (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5,
            int(round(self.width * s)), int(round(self.height * s)),
            self.dist.copy(),
        )

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps({
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "dist": self.dist.tolist(),
        }, indent=2))

    @classmethod
    def from_json(cls, path) -> "PinholeCamera":
        d = json.loads(Path(path).read_text())
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"],
                   np.asarray(d.get("dist", np.zeros(5))))

    # ------------------------------------------------------------- distortion
    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        """Apply lens distortion to (N, 2) normalised image-plane coordinates.

            r^2 = x^2 + y^2
            radial     = 1 + k1 r^2 + k2 r^4 + k3 r^6
            x_d = x*radial + 2 p1 x y + p2 (r^2 + 2 x^2)
            y_d = y*radial + p1 (r^2 + 2 y^2) + 2 p2 x y

        The radial term is a function of r alone, so it slides points along
        rays from the principal point -- it cannot rotate them.  The tangential
        term is what breaks that symmetry.
        """
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([x_d, y_d], axis=1)

    def _distortion_jacobian(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2, 2) Jacobian d(x_d, y_d)/d(x, y) of the distortion model.

        Needed for Newton's method below.  Writing it analytically rather than
        by finite differences keeps the inversion accurate to machine precision
        and is the same derivation pattern used later for the pose Jacobian in
        refinement.
        """
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        g = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3          # radial factor
        gr = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2**2           # d(g)/d(r^2)
        J = np.empty((len(xy), 2, 2))
        J[:, 0, 0] = g + 2.0 * x * x * gr + 2.0 * p1 * y + 6.0 * p2 * x
        J[:, 0, 1] = 2.0 * x * y * gr + 2.0 * p1 * x + 2.0 * p2 * y
        J[:, 1, 0] = 2.0 * x * y * gr + 2.0 * p1 * x + 2.0 * p2 * y
        J[:, 1, 1] = g + 2.0 * y * y * gr + 6.0 * p1 * y + 2.0 * p2 * x
        return J

    def undistort_normalized(self, xy_d: np.ndarray, iters: int = 10,
                             tol: float = 1e-12, return_converged: bool = False):
        """Invert :meth:`distort_normalized` by Newton's method.

        Brown-Conrady has no closed-form inverse -- it is a degree-7 polynomial
        system in (x, y) -- so the inverse is solved numerically:

            solve  distort(x) - x_d = 0   by   x <- x - J(x)^-1 (distort(x) - x_d)

        The cheaper alternative -- fixed-point iteration on
        ``x = (x_d - tangential) / radial``, which is what OpenCV's
        ``undistortPoints`` does by default -- is fine *inside* the image.
        Measured on the TUM freiburg1 intrinsics (k1=0.23, k2=-0.78, k3=0.92),
        it lands within 0.02 px of the true answer at the image corners after 5
        iterations and hits machine precision by 20.  It falls apart outside the
        image: at normalised radius ~1.4 it settles 0.31 off, because that is
        past the point where the radial polynomial stops being monotonic in r
        and the mapping is no longer injective at all.  Newton is used here
        anyway for three reasons: it converges quadratically (4 iterations to
        machine precision instead of 20), it still converges in that outer
        region, and deriving its Jacobian analytically is the same exercise as
        the pose Jacobian in refinement later.

        Where distortion is genuinely non-injective no solver can recover a
        unique inverse, which is why ``return_converged`` exists rather than
        trusting the output blindly.
        """
        xy_d = np.asarray(xy_d, dtype=float).reshape(-1, 2)
        if not self.has_distortion:
            out = xy_d.copy()
            return (out, np.ones(len(out), bool)) if return_converged else out

        xy = xy_d.copy()                       # distortion is small: x_d is a fine seed
        for _ in range(iters):
            residual = self.distort_normalized(xy) - xy_d
            if np.max(np.abs(residual)) < tol:
                break
            J = self._distortion_jacobian(xy)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            det = np.where(np.abs(det) < 1e-12, np.nan, det)   # fold singular into NaN
            # Closed-form 2x2 inverse times residual, vectorised over all points.
            dx = (J[:, 1, 1] * residual[:, 0] - J[:, 0, 1] * residual[:, 1]) / det
            dy = (-J[:, 1, 0] * residual[:, 0] + J[:, 0, 0] * residual[:, 1]) / det
            step = np.stack([dx, dy], axis=1)
            xy = xy - np.nan_to_num(step, nan=0.0)

        if return_converged:
            final = np.linalg.norm(self.distort_normalized(xy) - xy_d, axis=1)
            return xy, final < 1e-8
        return xy

    # ------------------------------------------------------------- projection
    def project(self, P_cam: np.ndarray, return_mask: bool = True):
        """Project (N, 3) points **in the camera frame** to (N, 2) pixels.

        Returns ``(uv, valid)`` where ``valid`` marks points that are in front
        of the camera (z > 0) and land inside the image.  Points behind the
        camera still produce numbers -- the perspective divide happily flips
        their sign and places them on the image plane as if they were in front.
        Silently keeping those is a classic source of phantom correspondences,
        which is why the mask is returned by default rather than offered.
        """
        P = np.asarray(P_cam, dtype=float).reshape(-1, 3)
        z = P[:, 2]
        in_front = z > 1e-9
        z_safe = np.where(in_front, z, 1.0)          # avoid divide-by-zero warnings
        xy = P[:, :2] / z_safe[:, None]
        if self.has_distortion:
            xy = self.distort_normalized(xy)
        uv = np.stack([self.fx * xy[:, 0] + self.cx,
                       self.fy * xy[:, 1] + self.cy], axis=1)
        if not return_mask:
            return uv
        inside = (
            (uv[:, 0] >= -0.5) & (uv[:, 0] <= self.width - 0.5)
            & (uv[:, 1] >= -0.5) & (uv[:, 1] <= self.height - 0.5)
        )
        return uv, in_front & inside & np.isfinite(uv).all(axis=1)

    def pixel_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> undistorted normalised image-plane coordinates (N, 2).

        This is K^-1 applied to the homogeneous pixel, followed by undistortion.
        It is the step that removes the camera's specific intrinsics from the
        measurement, leaving a purely geometric quantity -- which is why PnP and
        epipolar geometry are formulated on normalised coordinates.
        """
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        xy_d = np.stack([(uv[:, 0] - self.cx) / self.fx,
                         (uv[:, 1] - self.cy) / self.fy], axis=1)
        return self.undistort_normalized(xy_d)

    def pixel_to_ray(self, uv: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Pixels -> (N, 3) ray directions in the camera frame.

        A single pixel constrains a 3D point to a ray, not a point: depth is
        unobservable from one view.  Recovering it takes a second view
        (triangulation) or a known map (relocalization).
        """
        xy = self.pixel_to_normalized(uv)
        rays = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
        if normalize:
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        return rays

    def project_world(self, P_world: np.ndarray, T_cam_world, return_mask: bool = True):
        """Project world points given the **extrinsics** ``T_cam_world``.

        Passing ``T_world_cam`` here (the camera pose) gives a wrong answer with
        no error, so the argument is named for the exact transform it wants.
        """
        return self.project(T_cam_world.transform(P_world), return_mask=return_mask)

    def reprojection_error(self, P_world, uv_observed, T_cam_world) -> np.ndarray:
        """Per-point reprojection residual in pixels (N,).

        This scalar is the objective that PnP, RANSAC scoring, and bundle
        adjustment all minimise or threshold on.  Measuring it in pixels rather
        than metres is deliberate: it is the space the *measurement noise* lives
        in, so a fixed threshold means the same thing at 1 m and at 10 m.
        """
        uv_hat = self.project_world(P_world, T_cam_world, return_mask=False)
        return np.linalg.norm(uv_hat - np.asarray(uv_observed, float).reshape(-1, 2), axis=1)

    def fov_deg(self) -> tuple[float, float]:
        """(horizontal, vertical) field of view in degrees, ignoring distortion."""
        return (
            float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx)))),
            float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy)))),
        )
