"""Absolute Trajectory Error and Relative Pose Error.

The two metrics answer different questions and disagree on purpose.

**ATE** aligns the whole estimated trajectory to ground truth once, then
measures how far each camera position ends up from where it should be.  It is
a *global consistency* measure: it punishes accumulated drift and a map that
has bent out of shape.  It is the headline number for relocalization, because
relocalization is precisely the claim "I know where I am in this map".

**RPE** ignores global alignment entirely and compares motion over a fixed
interval -- how far did you travel between t and t+1s, versus how far did you
actually travel.  It is a *local accuracy* measure, insensitive to drift that
accumulated before the interval started.  It is the right number for odometry.

They come apart in both directions, which is why both get reported:

* A system that drifts slowly but smoothly has excellent RPE and terrible ATE.
* A system that jitters frame to frame but never drifts has good ATE and bad RPE.

Definitions follow Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM
Systems" (IROS 2012), so the numbers are comparable to published results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.trajectory import Trajectory, associate
from ..geometry.rotations import log_so3
from ..geometry.se3 import SE3
from .align import umeyama_alignment


def _stats(errors: np.ndarray) -> dict:
    """RMSE plus the distribution, because RMSE alone hides the failure mode.

    RMSE is dominated by the tail: a trajectory that is perfect for 95% of its
    length and catastrophically lost for 5% can report the same RMSE as one
    that is uniformly mediocre.  Median and max separate those two cases, and
    they are different engineering problems.
    """
    e = np.asarray(errors, float)
    if len(e) == 0:
        return {"rmse": float("nan"), "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "std": float(np.std(e)),
        "min": float(np.min(e)),
        "max": float(np.max(e)),
        "n": int(len(e)),
    }


@dataclass
class ATEResult:
    """Absolute trajectory error. Translation errors are metres.

    ``rotation_deg`` comes with a caveat that is easy to miss and produces
    confidently wrong conclusions.

    The alignment is fitted to **positions only** -- that is what Umeyama
    consumes and what the standard definition of ATE uses. That alignment
    rotation is then applied to the estimate's orientations too, because a
    trajectory has to be moved rigidly. So whenever the aligner rotates the
    trajectory to better fit the positions, it *introduces* an orientation
    offset that had nothing to do with the estimate's rotation accuracy.

    Measured on TUM fr1/xyz, injecting pure translation drift and zero
    rotation error: 2 cm/s of drift makes the aligner rotate by 14.22 deg, and
    the reported ATE rotation RMSE comes out as 14.220 deg. It is the alignment
    rotation, not an error in the estimate.

    So: read ``rotation_deg`` only alongside ``alignment_rotation_deg``. If
    they are close, the number says nothing about the estimate. **RPE rotation
    is the honest measure of orientation accuracy**, because RPE never aligns
    anything. This is also why the standard TUM ATE is translation-only.
    """

    translation: dict                      # rmse/mean/median/std/min/max/n, metres
    rotation_deg: dict                     # degrees -- see the caveat above
    alignment: SE3                         # transform applied to the estimate
    scale: float                           # 1.0 unless Sim(3) alignment was used
    errors: np.ndarray = field(repr=False)  # (N,) per-pose translation error
    timestamps: np.ndarray = field(repr=False)
    n_pairs: int = 0
    alignment_mode: str = "se3"

    @property
    def alignment_rotation_deg(self) -> float:
        """How far the alignment itself rotated the trajectory."""
        return float(np.degrees(np.linalg.norm(log_so3(self.alignment.R))))

    def summary(self) -> str:
        t, r = self.translation, self.rotation_deg
        out = (f"ATE ({self.alignment_mode}, scale={self.scale:.5f}, n={self.n_pairs})\n"
               f"  translation  rmse {t['rmse']*100:7.2f} cm   median {t['median']*100:7.2f} cm"
               f"   max {t['max']*100:7.2f} cm\n"
               f"  rotation     rmse {r['rmse']:7.3f} deg  median {r['median']:7.3f} deg"
               f"   max {r['max']:7.3f} deg")
        a = self.alignment_rotation_deg
        if a > 0.5 and abs(a - r["rmse"]) < 0.25 * max(r["rmse"], 1e-9):
            out += (f"\n  NOTE: alignment itself rotated {a:.2f} deg, which accounts for"
                    f"\n        the rotation figure above. Use RPE rotation instead.")
        return out


@dataclass
class RPEResult:
    """Relative pose error over a fixed interval."""

    translation: dict                      # metres per interval
    rotation_deg: dict                     # degrees per interval
    delta: float
    delta_unit: str
    errors_trans: np.ndarray = field(repr=False)
    errors_rot_deg: np.ndarray = field(repr=False)
    n_pairs: int = 0

    def summary(self) -> str:
        t, r = self.translation, self.rotation_deg
        unit = "s" if self.delta_unit == "s" else "frames"
        return (f"RPE (delta={self.delta:g} {unit}, n={self.n_pairs})\n"
                f"  translation  rmse {t['rmse']*100:7.2f} cm/{self.delta:g}{unit[0]}"
                f"   median {t['median']*100:7.2f} cm\n"
                f"  rotation     rmse {r['rmse']:7.3f} deg/{self.delta:g}{unit[0]}"
                f"   median {r['median']:7.3f} deg")


def _align_pairs(estimate: Trajectory, groundtruth: Trajectory,
                 max_diff: float, offset: float):
    pairs = associate(estimate.timestamps, groundtruth.timestamps, max_diff, offset)
    if len(pairs) == 0:
        raise ValueError(
            "no timestamp pairs within max_diff -- the two trajectories do not "
            "overlap in time. Check for a clock offset (pass `offset=`) or a "
            "seconds-vs-nanoseconds unit mismatch.")
    est = Trajectory(estimate.timestamps[pairs[:, 0]],
                     [estimate.poses[i] for i in pairs[:, 0]], estimate.name)
    gt = Trajectory(groundtruth.timestamps[pairs[:, 1]],
                    [groundtruth.poses[j] for j in pairs[:, 1]], groundtruth.name)
    return est, gt


def ate(estimate: Trajectory, groundtruth: Trajectory, align: str = "se3",
        max_diff: float = 0.02, offset: float = 0.0) -> ATEResult:
    """Absolute trajectory error after global alignment.

    Parameters
    ----------
    align : {"se3", "sim3", "none"}
        ``se3`` for a metric system (RGB-D, stereo, or relocalization into a
        metric map) -- this is the honest choice here, since a scale error is a
        real error.  ``sim3`` for monocular, where scale is unobservable in
        principle.  ``none`` when both trajectories are already in the same
        frame and you want to see the raw difference, which is the case when
        evaluating relocalization *into a known map*: there is no free gauge to
        absorb, and alignment would flatter the result.

    Returns
    -------
    ATEResult
        Translation errors in metres and rotation errors in degrees, plus the
        alignment that was applied, so the number can be reproduced. Read the
        ATEResult docstring before quoting the rotation figure -- it is
        contaminated by the alignment whenever the aligner had to rotate much.
    """
    if align not in ("se3", "sim3", "none"):
        raise ValueError(f"align must be 'se3', 'sim3' or 'none', got {align!r}")

    est, gt = _align_pairs(estimate, groundtruth, max_diff, offset)

    if align == "none":
        T_align, scale = SE3.identity("target", "source"), 1.0
    else:
        T_align, scale = umeyama_alignment(est.positions, gt.positions,
                                           with_scale=(align == "sim3"))
    aligned = est.transformed(T_align, scale)

    trans_err = np.linalg.norm(aligned.positions - gt.positions, axis=1)
    rot_err = np.array([
        np.degrees(np.linalg.norm(log_so3(G.R.T @ A.R)))
        for A, G in zip(aligned.poses, gt.poses)
    ])

    return ATEResult(
        translation=_stats(trans_err), rotation_deg=_stats(rot_err),
        alignment=T_align, scale=scale, errors=trans_err,
        timestamps=gt.timestamps, n_pairs=len(gt), alignment_mode=align)


def rpe(estimate: Trajectory, groundtruth: Trajectory, delta: float = 1.0,
        delta_unit: str = "s", max_diff: float = 0.02, offset: float = 0.0,
        scale: float = 1.0, pair_tol: float = 0.02) -> RPEResult:
    """Relative pose error over intervals of length ``delta``.

    For each pair of times (i, j) separated by roughly ``delta``, form the
    motion each trajectory claims to have made and compare them::

        E_ij = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j)

    where Q is ground truth and P the estimate.  ``Q_i^-1 Q_j`` is the true
    motion expressed in frame i; ``P_i^-1 P_j`` is the estimated motion in the
    estimate's own frame i.  Because both are *relative*, the arbitrary world
    frames cancel -- which is why RPE needs no alignment, unlike ATE.  A perfect
    estimate gives E = identity, so the error is the translation and rotation
    angle of E.

    Parameters
    ----------
    delta, delta_unit
        Interval length, in seconds (``"s"``) or in associated frames (``"f"``).
        The choice matters: 1 second is the conventional reporting unit
        (drift per second), while frame-based deltas depend on frame rate and
        are not comparable across datasets.
    scale
        Multiplies estimated translations before comparison.  For a monocular
        system, pass the scale from the Sim(3) ATE alignment -- otherwise RPE
        reports the unknown scale factor as if it were motion error.
    pair_tol
        How close to ``delta`` a pair's actual time separation must be, in
        seconds.  Pairs that cannot be formed are skipped rather than stretched.
    """
    if delta_unit not in ("s", "f"):
        raise ValueError(f"delta_unit must be 's' or 'f', got {delta_unit!r}")

    est, gt = _align_pairs(estimate, groundtruth, max_diff, offset)
    n = len(est)
    ts = gt.timestamps

    if delta_unit == "f":
        step = int(delta)
        if step < 1:
            raise ValueError("frame delta must be >= 1")
        pairs = [(i, i + step) for i in range(n - step)]
    else:
        pairs = []
        for i in range(n):
            j = int(np.searchsorted(ts, ts[i] + delta))
            for cand in (j - 1, j):
                if i < cand < n and abs((ts[cand] - ts[i]) - delta) <= pair_tol:
                    pairs.append((i, cand))
                    break

    if not pairs:
        raise ValueError(
            f"no pose pairs separated by {delta}{delta_unit} within {pair_tol}s. "
            "The trajectory may be shorter than delta, or too sparse.")

    trans, rot = [], []
    for i, j in pairs:
        # Relative motion in each trajectory's own frame i.
        gt_rel = gt.poses[i].inverse() @ gt.poses[j]
        est_rel = est.poses[i].inverse() @ est.poses[j]
        est_t = scale * est_rel.t
        E_R = gt_rel.R.T @ est_rel.R
        E_t = gt_rel.R.T @ (est_t - gt_rel.t)
        trans.append(np.linalg.norm(E_t))
        rot.append(np.degrees(np.linalg.norm(log_so3(E_R))))

    trans, rot = np.array(trans), np.array(rot)
    return RPEResult(translation=_stats(trans), rotation_deg=_stats(rot),
                     delta=delta, delta_unit=delta_unit,
                     errors_trans=trans, errors_rot_deg=rot, n_pairs=len(pairs))
