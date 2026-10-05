"""A timestamped sequence of camera poses, and TUM-format I/O.

A trajectory here is always ``T_world_cam`` -- the camera pose in the world
frame, so ``poses[i].t`` is literally where the camera was.  That is the
convention TUM ground truth publishes and the one the evaluation metrics
assume.  If you hand this class extrinsics (``T_cam_world``) instead, every
metric still returns a number; the number is just meaningless.  Hence the
frame labels on every pose built here.

TUM trajectory file format
--------------------------
Plain text, one pose per line, ``#`` starts a comment::

    timestamp tx ty tz qx qy qz qw

Timestamps are seconds since the Unix epoch, as floats.  Position is the
optical centre of the colour camera with respect to the world origin.  The
quaternion is **xyzw** -- scalar LAST -- which is the opposite of the wxyz
ordering used internally by ``anchor.geometry.rotations``.  Getting this
backwards yields a perfectly valid rotation that is simply the wrong one, with
no error anywhere, so the conversion goes through the explicitly-named
``quat_xyzw_to_rot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.rotations import exp_so3, log_so3, quat_xyzw_to_rot, rot_to_quat_xyzw
from ..geometry.se3 import SE3


@dataclass
class Trajectory:
    """Timestamps (N,) in seconds and the matching ``T_world_cam`` poses."""

    timestamps: np.ndarray
    poses: list[SE3]
    name: str = ""

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if len(self.timestamps) != len(self.poses):
            raise ValueError(
                f"{len(self.timestamps)} timestamps but {len(self.poses)} poses")
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing; sort before constructing")

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Trajectory(self.timestamps[i], self.poses[i], self.name)
        return self.timestamps[i], self.poses[i]

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) camera centres in world coordinates."""
        return np.array([T.t for T in self.poses])

    @property
    def rotations(self) -> np.ndarray:
        """(N, 3, 3) world-from-camera rotations."""
        return np.array([T.R for T in self.poses])

    @property
    def duration(self) -> float:
        return 0.0 if len(self) == 0 else float(self.timestamps[-1] - self.timestamps[0])

    def path_length(self) -> float:
        """Total distance travelled in metres.

        Useful as a denominator: 5 cm of drift over a 2 m path is a very
        different result from 5 cm over 100 m, and papers report both.
        """
        p = self.positions
        return 0.0 if len(p) < 2 else float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def transformed(self, T_new_world: SE3, scale: float = 1.0) -> "Trajectory":
        """Apply a similarity transform to the whole trajectory.

        Used to put an estimated trajectory into the ground-truth frame before
        computing ATE.  Both the camera *position* and its *orientation* move::

            R_new_cam = R_new_world @ R_world_cam
            t_new     = s * (R_new_world @ t_world) + t_new_world

        The scale multiplies the position only.  Scaling the rotation would be
        meaningless -- a rotation has no size.  This asymmetry is exactly why a
        monocular trajectory can be recovered up to an unknown scale factor
        while still having perfectly determined orientations.
        """
        R, t = T_new_world.R, T_new_world.t
        poses = [SE3(R @ P.R, scale * (R @ P.t) + t, "world", "cam") for P in self.poses]
        return Trajectory(self.timestamps.copy(), poses, self.name)

    # ------------------------------------------------------------------- I/O
    @classmethod
    def from_tum_file(cls, path, name: str = "") -> "Trajectory":
        """Read a TUM-format trajectory file (``timestamp tx ty tz qx qy qz qw``)."""
        path = Path(path)
        ts, poses = [], []
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"{path}:{lineno}: expected 8 fields "
                    f"(timestamp tx ty tz qx qy qz qw), got {len(parts)}: {raw!r}")
            v = [float(x) for x in parts]
            q = np.array(v[4:8])
            n = np.linalg.norm(q)
            if not (0.9 < n < 1.1):
                raise ValueError(f"{path}:{lineno}: quaternion norm {n:.4f} is not ~1; "
                                 "wrong column order or corrupt file")
            ts.append(v[0])
            poses.append(SE3(quat_xyzw_to_rot(q), np.array(v[1:4]), "world", "cam"))
        order = np.argsort(ts, kind="stable")   # mocap files are usually sorted, but not guaranteed
        return cls(np.array(ts)[order], [poses[i] for i in order], name or path.stem)

    def to_tum_file(self, path) -> None:
        """Write in TUM format, so results can be fed to the official evaluation scripts."""
        lines = ["# timestamp tx ty tz qx qy qz qw"]
        for t, T in zip(self.timestamps, self.poses):
            qx, qy, qz, qw = rot_to_quat_xyzw(T.R)
            lines.append(f"{t:.6f} {T.t[0]:.9f} {T.t[1]:.9f} {T.t[2]:.9f} "
                         f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
        Path(path).write_text("\n".join(lines) + "\n")


def associate(stamps_a: np.ndarray, stamps_b: np.ndarray,
              max_diff: float = 0.02, offset: float = 0.0) -> np.ndarray:
    """Match two timestamp streams into (i, j) index pairs, greedily by closeness.

    Different sensors are not synchronised: TUM colour frames arrive at ~30 Hz
    while motion-capture ground truth is ~100 Hz, and no timestamp appears in
    both files.  Every evaluation therefore starts by deciding which ground
    truth pose "is" the one for a given frame.

    The rule here is the standard TUM one: consider all pairs within
    ``max_diff`` seconds, take the closest pair, remove both entries, repeat.
    That greedy pass guarantees a one-to-one matching -- important, because
    nearest-neighbour matching in one direction can map several frames onto the
    same ground truth pose and quietly duplicate it in the error statistics.

    ``max_diff`` defaults to 0.02 s, TUM's own default.  At a 30 Hz frame rate
    that is 60% of a frame interval, and any pair further apart than that is
    matching a frame to ground truth from a genuinely different instant --
    during fast motion, that timing error becomes position error which is then
    reported as trajectory error.

    ``offset`` is added to ``stamps_b`` before matching, for the case where two
    devices' clocks are offset by a constant.
    """
    a = np.asarray(stamps_a, float)
    b = np.asarray(stamps_b, float) + offset
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), dtype=int)

    # All candidate pairs within the window, sorted by |dt|. O(N*M) is fine for
    # trajectory-sized data (thousands of entries) and is exactly reproducible.
    diff = np.abs(a[:, None] - b[None, :])
    ii, jj = np.nonzero(diff <= max_diff)
    if len(ii) == 0:
        return np.zeros((0, 2), dtype=int)
    order = np.argsort(diff[ii, jj], kind="stable")

    used_a, used_b, pairs = set(), set(), []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    pairs.sort()
    return np.array(pairs, dtype=int)


def interpolate_pose(traj: Trajectory, t: float, max_gap: float = 0.05) -> SE3 | None:
    """``T_world_cam`` at time ``t``, interpolated between the bracketing samples.

    Rotation moves along the geodesic between the two samples at constant
    angular rate, ``R = R0 exp(a log(R0^T R1))``; that is SLERP written on
    matrices. Translation is linear. For 10 ms mocap intervals the difference
    from a full SE(3) geodesic (``T0 exp(a log(T0^-1 T1))``, which curves the
    translation path) is far below the mocap's own noise.

    Returns None outside the trajectory's time span or across a gap longer
    than ``max_gap`` seconds -- interpolating across a mocap dropout invents a
    pose the camera may never have had.
    """
    ts = traj.timestamps
    if len(ts) == 0 or t < ts[0] or t > ts[-1]:
        return None
    k = int(np.searchsorted(ts, t, side="right"))
    if k == len(ts):                       # t == last timestamp
        return traj.poses[-1]
    t0, t1 = ts[k - 1], ts[k]
    if t1 - t0 > max_gap:
        return None
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    P0, P1 = traj.poses[k - 1], traj.poses[k]
    R = P0.R @ exp_so3(a * log_so3(P0.R.T @ P1.R))
    return SE3(R, (1.0 - a) * P0.t + a * P1.t, P0.to_frame, P0.from_frame)


def camera_speed(traj: Trajectory, t: float, h: float = 0.02) -> tuple[float, float] | None:
    """(linear speed m/s, angular speed rad/s) at time ``t``, by central difference over ±h.

    Angular speed is what limits how far ground truth can be trusted at a
    given frame: any timing error dt between image and pose becomes a
    rotation error of ``omega * dt``, which moves every pixel by about
    ``f * omega * dt`` regardless of depth. At fr1's f = 517 px and 30 deg/s,
    a 4 ms timing error is already 1 px.
    """
    a, b = interpolate_pose(traj, t - h), interpolate_pose(traj, t + h)
    if a is None or b is None:
        return None
    lin = float(np.linalg.norm(b.t - a.t) / (2 * h))
    ang = float(np.linalg.norm(log_so3(a.R.T @ b.R)) / (2 * h))
    return lin, ang
