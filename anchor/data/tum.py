"""Loader for the TUM RGB-D SLAM Dataset and Benchmark.

Layout of a sequence directory::

    rgbd_dataset_freiburg1_xyz/
        rgb.txt          # "timestamp filename", 640x480 8-bit PNG, ~30 Hz
        depth.txt        # "timestamp filename", 640x480 16-bit PNG, ~30 Hz
        groundtruth.txt  # "timestamp tx ty tz qx qy qz qw", motion capture, ~100 Hz
        rgb/ depth/

Three separate clocks, none of them aligned, so loading a sequence *is*
mostly an association problem -- see :func:`anchor.data.trajectory.associate`.

Depth encoding: 16-bit PNG, value / 5000 = metres, 0 means no measurement.
Missing data is common and not an error -- the Kinect's projector sees nothing
in shadows, on specular surfaces, past ~4 m, or through anything transparent.
Treating 0 as "a point at the origin" puts a cluster of fake points right at the
camera centre, which is a real and quiet way to poison a map.

The depth images shipped by TUM are already registered to the colour frame, so
colour intrinsics are the right ones to unproject with.

Ground-truth frame convention
-----------------------------
TUM documents the pose as "the position of the optical center of the color
camera with respect to the world origin", i.e. ``T_world_cam`` -- the camera
pose, not the extrinsics.  Secondary sources describe this inconsistently, so
:func:`verify_groundtruth_convention` checks it empirically against the data
rather than trusting the documentation or this comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.camera import PinholeCamera
from ..geometry.se3 import SE3
from .trajectory import Trajectory, associate

# Per-camera intrinsics. TUM calibrated the three Freiburg cameras separately;
# using the wrong set is a systematic error of several pixels, which turns into
# a systematic pose bias that no amount of good matching will fix.
#
# Values transcribed from the ORB-SLAM3 TUM1/TUM2/TUM3 example configs, which
# is the deliberate choice: ORB-SLAM3 is the benchmark baseline for this
# project, and using identical intrinsics keeps the comparison honest. The TUM
# wiki's own calibration page did not render machine-readably when checked.
_TUM_INTRINSICS = {
    #             fx          fy          cx          cy      k1        k2         p1         p2        k3
    "ros":  (525.000000, 525.000000, 319.500000, 239.500000,  0.0,      0.0,       0.0,       0.0,      0.0),
    "fr1":  (517.306408, 516.469215, 318.643040, 255.313989,  0.262383, -0.953104, -0.005358, 0.002628, 1.163314),
    "fr2":  (520.908620, 521.007327, 325.141442, 249.701764,  0.231222, -0.784899, -0.003257, -0.000105, 0.917205),
    "fr3":  (535.400000, 539.200000, 320.100000, 247.600000,  0.0,      0.0,       0.0,       0.0,      0.0),
}

# Depth scale, in units per metre. Note fr2 is NOT 5000: TUM published a
# per-camera depth correction factor and ORB-SLAM3's TUM2 config folds it in as
# 5208. Using 5000 for fr2 makes every depth ~4% too large, which shows up as a
# consistent scale error in the reconstructed map -- exactly the kind of bug
# that survives all the way to the metrics table looking like "our system has
# 4% scale drift".
_DEPTH_SCALE = {"ros": 5000.0, "fr1": 5000.0, "fr2": 5208.0, "fr3": 5000.0}


def tum_camera(category: str = "fr1") -> tuple[PinholeCamera, float]:
    """Return ``(camera, depth_scale)`` for ``"ros"``, ``"fr1"``, ``"fr2"`` or ``"fr3"``."""
    key = category.lower()
    if key not in _TUM_INTRINSICS:
        raise KeyError(f"unknown TUM camera {category!r}; expected one of {list(_TUM_INTRINSICS)}")
    fx, fy, cx, cy, *dist = _TUM_INTRINSICS[key]
    return PinholeCamera(fx, fy, cx, cy, 640, 480, dist), _DEPTH_SCALE[key]


def detect_category(root) -> str:
    """Infer the Freiburg camera from a sequence directory name."""
    name = Path(root).name.lower()
    for key in ("freiburg1", "freiburg2", "freiburg3"):
        if key in name:
            return "fr" + key[-1]
    raise ValueError(
        f"cannot tell which camera {Path(root).name!r} used; pass category= explicitly. "
        "Guessing here would silently use intrinsics off by several pixels.")


def read_file_list(path) -> tuple[np.ndarray, list[str]]:
    """Read a TUM ``timestamp filename`` index file (rgb.txt / depth.txt)."""
    stamps, names = [], []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected 'timestamp filename', got {raw!r}")
        stamps.append(float(parts[0]))
        names.append(parts[1])
    return np.array(stamps), names


@dataclass
class TUMFrame:
    """One RGB(-D) frame with the ground-truth pose associated to it."""

    timestamp: float                  # the RGB timestamp; depth/pose are within max_diff
    rgb_path: Path
    depth_path: Path | None = None
    T_world_cam: SE3 | None = None    # ground-truth camera pose, if available
    depth_timestamp: float | None = None
    pose_timestamp: float | None = None

    @property
    def dt_depth(self) -> float:
        """Seconds between the colour and depth exposures -- nonzero, always."""
        return 0.0 if self.depth_timestamp is None else abs(self.depth_timestamp - self.timestamp)


class TUMDataset:
    """A loaded TUM RGB-D sequence: frames, intrinsics, and ground truth."""

    def __init__(self, root, category: str | None = None, max_diff: float = 0.02,
                 require_depth: bool = True, require_pose: bool = True):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"no such sequence directory: {self.root}")
        self.category = category or detect_category(self.root)
        self.camera, self.depth_scale = tum_camera(self.category)

        rgb_ts, rgb_names = read_file_list(self.root / "rgb.txt")

        depth_ts, depth_names = (np.zeros(0), [])
        if (self.root / "depth.txt").exists():
            depth_ts, depth_names = read_file_list(self.root / "depth.txt")

        self.groundtruth: Trajectory | None = None
        if (self.root / "groundtruth.txt").exists():
            self.groundtruth = Trajectory.from_tum_file(
                self.root / "groundtruth.txt", name=self.root.name)

        # Associate depth and ground truth onto the colour frames independently.
        depth_for = dict(associate(rgb_ts, depth_ts, max_diff)) if len(depth_ts) else {}
        pose_for = (dict(associate(rgb_ts, self.groundtruth.timestamps, max_diff))
                    if self.groundtruth is not None else {})

        self.frames: list[TUMFrame] = []
        for i, (t, name) in enumerate(zip(rgb_ts, rgb_names)):
            j, k = depth_for.get(i), pose_for.get(i)
            if require_depth and j is None:
                continue
            if require_pose and k is None:
                continue
            self.frames.append(TUMFrame(
                timestamp=float(t),
                rgb_path=self.root / name,
                depth_path=(self.root / depth_names[j]) if j is not None else None,
                T_world_cam=self.groundtruth.poses[k] if k is not None else None,
                depth_timestamp=float(depth_ts[j]) if j is not None else None,
                pose_timestamp=float(self.groundtruth.timestamps[k]) if k is not None else None,
            ))

        self.n_rgb_total = len(rgb_ts)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i) -> TUMFrame:
        return self.frames[i]

    def trajectory(self) -> Trajectory:
        """Ground-truth poses for the frames that survived association."""
        keep = [f for f in self.frames if f.T_world_cam is not None]
        return Trajectory(np.array([f.timestamp for f in keep]),
                          [f.T_world_cam for f in keep], self.root.name)

    # ------------------------------------------------------------------ images
    def load_rgb(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB image (OpenCV loads BGR; converted here)."""
        import cv2
        img = cv2.imread(str(self.frames[i].rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read {self.frames[i].rgb_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_gray(self, i: int) -> np.ndarray:
        import cv2
        img = cv2.imread(str(self.frames[i].rgb_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read {self.frames[i].rgb_path}")
        return img

    def load_depth(self, i: int) -> np.ndarray:
        """(H, W) float32 depth in **metres**, with 0 replaced by NaN.

        NaN rather than 0 is deliberate: 0 is a valid-looking number that
        propagates silently through arithmetic, while NaN makes any accidental
        use of a missing measurement show up immediately.
        """
        import cv2
        path = self.frames[i].depth_path
        if path is None:
            raise ValueError(f"frame {i} has no associated depth image")
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"could not read {path}")
        if raw.dtype != np.uint16:
            raise ValueError(f"{path}: expected 16-bit depth PNG, got {raw.dtype}")
        depth = raw.astype(np.float32) / self.depth_scale
        depth[raw == 0] = np.nan
        return depth

    def unproject(self, i: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Back-project frame ``i``'s depth into 3D. Returns ``(points_cam, uv)``.

        Each valid pixel plus its depth determines a point exactly: the pixel
        gives a ray (undistorted, so the geometry is right), and depth gives how
        far along it to stop.  ``z`` in the depth image is the *depth along the
        optical axis*, not the distance along the ray, so the ray is scaled to
        unit z before multiplying -- using a unit-length ray instead makes every
        point too far away by a factor of ``1/cos(angle from axis)``, up to about
        11% at the corners of a 60-degree lens.
        """
        depth = self.load_depth(i)
        vs, us = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        uv = np.stack([us.ravel(), vs.ravel()], axis=1).astype(float)
        z = depth[vs.ravel(), us.ravel()]

        good = np.isfinite(z)
        uv, z = uv[good], z[good]

        rays = self.camera.pixel_to_ray(uv, normalize=False)   # z-component is 1
        return rays * z[:, None], uv


def verify_groundtruth_convention(dataset: "TUMDataset", i: int = 0, j: int = 30,
                                  stride: int = 4) -> dict:
    """Empirically decide whether ground truth is ``T_world_cam`` or its inverse.

    The documentation says one thing and secondary sources say another, and
    getting it backwards produces a trajectory that looks plausible and is
    wrong. So test it against the data instead of believing anyone.

    The test is dense geometric consistency between two views of the same rigid
    scene:

    1. Unproject frame i's depth image into 3D points in camera i.
    2. Push them into the world using the candidate convention, then into
       camera j, and project to pixels.
    3. Compare the depth those points *predict* at each pixel of frame j
       against the depth frame j actually *measured* there.

    Under the correct convention the two agree to within sensor noise. Under
    the wrong one, the relative motion between the frames is wrong and the
    predicted depths are visibly off.

    Note the discriminating power depends on how much the camera moved between
    i and j: with a nearly stationary camera both conventions give a nearly
    identity relative transform and the test weakens. Use a gap of at least a
    second, and read ``ratio`` -- anything below ~2 is not a conclusive answer.

    Returns
    -------
    dict
        Per-hypothesis median depth residual in metres and in-view fraction,
        plus ``verdict`` and ``ratio`` (how many times worse the loser is).
    """
    depth_j = dataset.load_depth(j)
    P_i, _ = dataset.unproject(i, stride=stride)
    T_i, T_j = dataset.frames[i].T_world_cam, dataset.frames[j].T_world_cam
    if T_i is None or T_j is None:
        raise ValueError("both frames need a ground-truth pose")

    h, w = depth_j.shape
    out: dict = {}
    for label, invert in (("as_T_world_cam", False), ("as_T_cam_world", True)):
        A, B = (T_i.inverse(), T_j.inverse()) if invert else (T_i, T_j)
        P_j = B.inverse().transform(A.transform(P_i))      # world -> camera j
        uv, ok = dataset.camera.project(P_j)

        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)

        if ok.sum() < 100:
            out[label] = {"median_depth_residual_m": float("inf"), "in_view_frac": float(ok.mean())}
            continue

        measured = depth_j[v[ok], u[ok]]
        predicted = P_j[ok, 2]
        valid = np.isfinite(measured)
        residual = np.abs(predicted[valid] - measured[valid])
        out[label] = {
            "median_depth_residual_m": float(np.median(residual)),
            "in_view_frac": float(ok.mean()),
            "n_compared": int(valid.sum()),
        }

    a = out["as_T_world_cam"]["median_depth_residual_m"]
    b = out["as_T_cam_world"]["median_depth_residual_m"]
    out["verdict"] = "as_T_world_cam" if a < b else "as_T_cam_world"
    out["ratio"] = float(max(a, b) / max(min(a, b), 1e-12))
    return out
