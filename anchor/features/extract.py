"""Keypoint detection, description, and the Features container.

Frames and units
----------------
* ``uv``: pixels, OpenCV convention (centre of the top-left pixel is (0, 0)),
  **as observed** -- i.e. still distorted. Keep these for drawing, for the
  labeller, and for anything that compares against raw images.
* ``xn``: normalized, **undistorted** image coordinates ``(X/Z, Y/Z)`` in the
  camera frame (OpenCV axes: x right, y down, z forward). Dimensionless. These
  are what geometry consumes: bearing vectors for PnP, epipolar checks,
  triangulation. They come from ``PinholeCamera.pixel_to_normalized`` (your
  Step 1 Newton inverse), not from ``cv2.undistortPoints``.
* ``desc``: SIFT is float32 (N, 128), compared with L2. ORB would be uint8
  (N, 32) compared with Hamming -- a different metric, so a different matcher.

Calling OpenCV detectors/descriptors here is allowed (ARCHITECTURE.md 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.camera import PinholeCamera


@dataclass
class Features:
    """Keypoints of one image. Every array has one row per keypoint."""

    uv: np.ndarray          # (N, 2) float64, pixels, distorted (as observed)
    xn: np.ndarray          # (N, 2) float64, normalized undistorted coords
    desc: np.ndarray        # (N, D) float32 (SIFT/RootSIFT) or uint8 (ORB)
    scale: np.ndarray       # (N,) float64, keypoint size in pixels (cv2.KeyPoint.size)
    angle: np.ndarray       # (N,) float64, orientation in DEGREES as OpenCV reports it
    response: np.ndarray    # (N,) float64, detector strength
    kind: str = "sift"      # "sift" | "rootsift" | "orb" -- decides the distance metric

    def __len__(self) -> int:
        return len(self.uv)

    def subset(self, idx) -> "Features":
        """Features restricted to ``idx`` (index array or boolean mask), same order."""
        raise NotImplementedError


def detect_and_describe(gray: np.ndarray, n_features: int = 2000, detector: str = "sift"):
    """Run an OpenCV detector + descriptor on a grayscale uint8 image.

    Returns ``(uv, desc, scale, angle, response)`` as NumPy arrays, ``uv`` in
    pixels (OpenCV convention, distorted). ``desc`` is ``None``-safe: an image
    with no keypoints must return empty arrays with the right shapes
    ((0, 2), (0, D), ...), not ``None`` -- OpenCV returns ``None`` there.
    """
    raise NotImplementedError


def rootsift(desc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """SIFT -> RootSIFT (Arandjelovic & Zisserman, CVPR 2012).

    L1-normalize each row, then take the element-wise square root. After this,
    the squared L2 distance between two descriptors equals
    ``2 - 2 * sum_k sqrt(a_k b_k)`` with ``a, b`` the L1-normalized originals:
    a function of the Hellinger kernel on the original histograms. Returns float32 (N, D) with unit L2 norm per row.

    Decide what an all-zero descriptor becomes (``eps`` guards the division).
    """
    raise NotImplementedError


def extract(gray: np.ndarray, camera: PinholeCamera, n_features: int = 2000,
            detector: str = "sift", root_sift: bool = False) -> Features:
    """Detect, describe, and compute both coordinate sets for one image.

    ``camera`` must be the intrinsics for *this* image's resolution. Fills
    ``xn`` with ``camera.pixel_to_normalized(uv)``.
    """
    raise NotImplementedError
