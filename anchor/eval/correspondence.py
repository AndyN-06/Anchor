"""Ground-truth labels for feature matches, from depth and known poses.

What this is for
----------------
A matcher outputs pairs ``(keypoint i in image A, keypoint j in image B)``. It
can't tell which pairs are right: descriptor distance is only a proxy. To
measure a matcher -- precision, recall, the effect of a ratio threshold --
every match needs an independent verdict. This module gives one, using
information the matcher never sees: A's depth image and the ground-truth
relative pose.

The geometry
------------
For keypoint ``u_a`` in A with measured depth ``z``::

    P_a  = z * ray(u_a)                    # 3D point in camera A, metres
    P_b  = T_camB_camA * P_a               # same point in camera B
    u_b* = project(P_b)                    # where it MUST appear in B, pixels

If the matcher paired ``u_a`` with ``u_b``, the match is correct when
``||u_b - u_b*|| <= thresh_px``.

That is the definition of a correct match used for the whole project
(ARCHITECTURE.md section 4.3 #6): a correspondence is correct if it is
consistent with the true pose, *not* if the descriptors look alike. It is also
the inlier test RANSAC will apply in Step 4, with the estimated pose in place
of the true one. So these labels say exactly how many inliers RANSAC has to
find.

Three verdicts, not two
-----------------------
``CORRECT``, ``WRONG`` and ``UNKNOWN``. A keypoint without a trustworthy depth
gets ``UNKNOWN``, and it is excluded from precision and recall rather than
guessed at. Keypoints are detected at corners and blobs, which are often depth
edges, where a pixel's depth can belong to the foreground or the background.
Counting those as wrong would blame the matcher for the depth sensor;
counting them as correct would be inventing data. The fraction excluded is
reported, because it is a bias: the labelled subset is the
depth-reliable subset, which leans towards surfaces and away from object
outlines.

How good the labels are on TUM (measured, scripts/label_tum_pair.py)
--------------------------------------------------------------------
The labeller has its own error: the predicted pixel is off by depth noise,
ground-truth pose error and image/pose timing. On fr1/xyz, with OpenCV SIFT
matches as a probe, correct matches do not sit at ~0.3 px (SIFT's own
localisation) but are spread over 1-3 px, with a *systematic* per-pair shift
of 1-3 px in a random direction. That shift is a ground-truth pose error, and
it scales with camera angular speed:

* both frames under 15 deg/s: median per-pair shift 1.0 px (5-frame gap),
  rising to ~3 px at a 60-frame gap
* both frames over 30 deg/s: 2.5-3 px, and precision at 3 px collapses
  (0.36 vs 0.82 for the same matcher at a 5-frame gap)

It isn't a depth-scale error (scaling depth by 0.90-1.10 helps some pairs and
hurts others) and a single global clock offset only reduces it slightly (best
-7.5 ms: 2.69 -> 2.24 px). Per-frame timing jitter, rolling shutter, or mocap
hand-eye calibration error are plausible causes; none is verified.

Consequences: use pairs where both frames are slow (``camera_speed``), keep
gaps short, use a 3-5 px threshold, and treat absolute precision on TUM as
biased low. Comparisons between matchers on the *same* pairs are still valid.
The synthetic tests remain the exact benchmark.

Frames and units: pixels in the OpenCV convention (centre of the top-left
pixel is (0, 0)), distorted as observed; depth in metres along the optical
axis, NaN where missing; ``T_camB_camA`` maps points from camera A's frame into
camera B's, metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.camera import PinholeCamera
from ..geometry.se3 import SE3

# Match verdicts.
CORRECT, WRONG, UNKNOWN = 1, 0, -1

# Why a keypoint in A can or cannot be transferred into B.
OK = 0
NO_DEPTH_A = 1        # depth missing at the keypoint (or in its 3x3 window)
DEPTH_EDGE_A = 2      # depth jumps inside the 3x3 window: fore/background ambiguous
BEHIND_B = 3          # the 3D point is behind camera B
OUT_OF_VIEW_B = 4     # projects outside image B
OCCLUDED_B = 5        # B's depth map shows something in front of the point
NO_DEPTH_B = 6        # can't check occlusion: B has no depth there

REASON_NAMES = {OK: "ok", NO_DEPTH_A: "no_depth_a", DEPTH_EDGE_A: "depth_edge_a",
                BEHIND_B: "behind_b", OUT_OF_VIEW_B: "out_of_view_b",
                OCCLUDED_B: "occluded_b", NO_DEPTH_B: "no_depth_b"}


@dataclass
class PairGeometry:
    """Everything needed to transfer a pixel from image A into image B."""

    camera: PinholeCamera          # same camera for A and B (TUM: one Kinect per sequence)
    T_camB_camA: SE3               # metres; maps points in A's frame into B's
    depth_a: np.ndarray            # (H, W) metres along z, NaN = missing
    depth_b: np.ndarray | None = None   # optional; enables the occlusion check

    def __post_init__(self) -> None:
        T = self.T_camB_camA
        # If frame labels are present, catch the inverted-transform bug here
        # rather than as a precision of ~0 that looks like a bad matcher.
        if T.to_frame is not None and T.from_frame is not None:
            if T.to_frame == T.from_frame:
                raise ValueError(f"T_camB_camA is labelled T_{T.to_frame}_{T.from_frame}: "
                                 "A and B must be different frames")


@dataclass
class Transfer:
    """Keypoints of A transferred into B. All arrays have one row per keypoint."""

    P_a: np.ndarray        # (N, 3) points in camera A, metres; NaN where depth unusable
    uv_b: np.ndarray       # (N, 2) predicted pixel in B; NaN where not computable
    z_b: np.ndarray        # (N,)  predicted depth in B, metres
    reason: np.ndarray     # (N,)  OK or one of the failure codes above

    @property
    def has_depth(self) -> np.ndarray:
        """Keypoints whose verdict can be decided at all."""
        return np.isfinite(self.P_a).all(axis=1)


def _window_depth(depth: np.ndarray, uv: np.ndarray, edge_tol: float):
    """Depth at the nearest pixel, plus whether that value can be trusted.

    Nearest pixel, not bilinear interpolation. Across a depth edge, bilinear
    interpolation averages a foreground and a background depth and produces a
    point floating in empty space between them, belonging to neither surface.

    A 3x3 window then guards the edge cases: any hole or a relative depth
    spread above ``edge_tol`` in the window means the keypoint sits on (or
    within a pixel of) a depth discontinuity. Which side it belongs to is
    ambiguous, and after the colour-depth timing offset (up to ~17 ms on TUM)
    the depth image may be shifted by a pixel or two relative to the colour
    image anyway.
    """
    h, w = depth.shape
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    inside = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)

    z = np.full(len(uv), np.nan)
    reason = np.full(len(uv), NO_DEPTH_A)
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return z, reason

    du, dv = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    win = depth[v[idx, None] + dv.ravel()[None, :], u[idx, None] + du.ravel()[None, :]]  # (n, 9)
    centre = win[:, 4]

    holes = ~np.isfinite(win).all(axis=1)
    spread = np.full(len(idx), np.inf)
    full = win[~holes]
    spread[~holes] = (full.max(axis=1) - full.min(axis=1)) / full.min(axis=1)
    edge = ~holes & (spread > edge_tol)

    good = ~holes & ~edge
    z[idx[good]] = centre[good]
    reason[idx[good]] = OK
    reason[idx[edge]] = DEPTH_EDGE_A
    return z, reason


def transfer_keypoints(uv_a: np.ndarray, geom: PairGeometry, edge_tol: float = 0.05,
                       occlusion_tol: float = 0.05) -> Transfer:
    """Predict where each keypoint of A must appear in B under the true pose.

    Steps, per keypoint:

    1. **Lift.** Read A's depth ``z`` at the keypoint (see :func:`_window_depth`)
       and scale the pixel's ray to it. The ray comes from the undistorted
       normalized coordinate with ``r_z = 1``, so ``z * r`` is the point --
       depth is z, not range.
    2. **Move.** ``P_b = T_camB_camA * P_a``. A pure change of coordinates; the
       point itself doesn't move.
    3. **Project** into B with B's (same) intrinsics and distortion.
    4. **Occlusion** (only if ``depth_b`` is given). If B measured something
       clearly *closer* than ``z_b`` at that pixel, a surface in front of the
       point hides it in B. ``occlusion_tol`` is relative, because Kinect depth
       noise grows with distance.

    Steps 3-4 only affect :func:`ground_truth_partners`. A match is still
    labelled by reprojection alone -- see :func:`label_matches`.
    """
    uv_a = np.asarray(uv_a, float).reshape(-1, 2)
    cam = geom.camera
    n = len(uv_a)

    z_a, reason = _window_depth(geom.depth_a, uv_a, edge_tol)
    P_a = np.full((n, 3), np.nan)
    ok = reason == OK
    if ok.any():
        P_a[ok] = cam.pixel_to_ray(uv_a[ok], normalize=False) * z_a[ok, None]

    uv_b = np.full((n, 2), np.nan)
    z_b = np.full(n, np.nan)
    if ok.any():
        P_b = geom.T_camB_camA.transform(P_a[ok])
        proj, valid = cam.project(P_b)
        idx = np.nonzero(ok)[0]
        z_b[idx] = P_b[:, 2]
        behind = P_b[:, 2] <= 1e-9
        uv_b[idx[~behind]] = proj[~behind]
        reason[idx[behind]] = BEHIND_B
        reason[idx[~behind & ~valid]] = OUT_OF_VIEW_B

        if geom.depth_b is not None:
            vis = idx[valid]
            h, w = geom.depth_b.shape
            u = np.clip(np.rint(uv_b[vis, 0]).astype(int), 0, w - 1)
            v = np.clip(np.rint(uv_b[vis, 1]).astype(int), 0, h - 1)
            measured = geom.depth_b[v, u]
            no_d = ~np.isfinite(measured)
            with np.errstate(invalid="ignore"):
                hidden = measured < z_b[vis] * (1.0 - occlusion_tol)
            reason[vis[no_d]] = NO_DEPTH_B
            reason[vis[~no_d & hidden]] = OCCLUDED_B

    return Transfer(P_a=P_a, uv_b=uv_b, z_b=z_b, reason=reason)


@dataclass
class MatchLabels:
    labels: np.ndarray       # (M,) CORRECT / WRONG / UNKNOWN
    error_px: np.ndarray     # (M,) distance from predicted to matched pixel; NaN if UNKNOWN
    thresh_px: float

    @property
    def n_correct(self) -> int:
        return int((self.labels == CORRECT).sum())

    @property
    def n_wrong(self) -> int:
        return int((self.labels == WRONG).sum())

    @property
    def n_unknown(self) -> int:
        return int((self.labels == UNKNOWN).sum())

    @property
    def precision(self) -> float:
        """Fraction of decidable matches that are correct. NaN if none are decidable."""
        d = self.n_correct + self.n_wrong
        return self.n_correct / d if d else float("nan")


def _check_matches(matches, n_a, n_b) -> np.ndarray:
    m = np.asarray(matches, dtype=int).reshape(-1, 2)
    if len(m) and (m[:, 0].min() < 0 or m[:, 0].max() >= n_a
                   or m[:, 1].min() < 0 or m[:, 1].max() >= n_b):
        raise IndexError("match indices out of range for the given keypoints")
    if len(np.unique(m[:, 0])) != len(m):
        # A keypoint of A matched twice can be counted correct twice, which
        # inflates recall past what is achievable. The matcher must not do it.
        raise ValueError("matches assign the same keypoint of A more than once")
    return m


def label_matches(uv_a, uv_b, matches, geom: PairGeometry, thresh_px: float = 3.0,
                  transfer: Transfer | None = None, **transfer_kw) -> MatchLabels:
    """Label each match CORRECT, WRONG or UNKNOWN.

    ``matches`` is (M, 2) integer index pairs ``(i into uv_a, j into uv_b)``.

    * UNKNOWN: keypoint i of A has no trustworthy depth.
    * WRONG: the point is behind camera B, or the matched pixel is more than
      ``thresh_px`` from where the point must appear.
    * CORRECT: within ``thresh_px``.

    Occlusion deliberately does not change a label. If the A-point is hidden
    in B and the matcher picked a keypoint at exactly its projected pixel, that
    2D-3D pair is still consistent with the true pose, so it is a genuine
    inlier for PnP. For two-view triangulation it also triangulates back to
    the A-point. Labelling by reprojection matches what Steps 4 and 6 need.

    A projection outside image B needs no special case: every keypoint of B is
    inside the image, so the error is large and the label is WRONG.

    ``thresh_px`` is not free. It must exceed the labeller's own error floor:
    depth noise, ground-truth pose error and the colour-depth timing offset.
    Measure that floor on real data (``scripts/label_tum_pair.py``) before
    choosing it.
    """
    uv_a = np.asarray(uv_a, float).reshape(-1, 2)
    uv_b = np.asarray(uv_b, float).reshape(-1, 2)
    m = _check_matches(matches, len(uv_a), len(uv_b))
    tr = transfer if transfer is not None else transfer_keypoints(uv_a, geom, **transfer_kw)

    i, j = m[:, 0], m[:, 1]
    labels = np.full(len(m), UNKNOWN)
    err = np.full(len(m), np.nan)

    decidable = tr.has_depth[i]
    behind = tr.reason[i] == BEHIND_B
    err[decidable & ~behind] = np.linalg.norm(
        uv_b[j[decidable & ~behind]] - tr.uv_b[i[decidable & ~behind]], axis=1)
    err[behind] = np.inf

    labels[decidable] = np.where(err[decidable] <= thresh_px, CORRECT, WRONG)
    return MatchLabels(labels=labels, error_px=err, thresh_px=thresh_px)


@dataclass
class Partners:
    """For each keypoint of A, the keypoint of B it should match, if any."""

    partner: np.ndarray      # (N,) index into uv_b, or -1
    distance_px: np.ndarray  # (N,) distance from prediction to that partner; NaN if none
    reason: np.ndarray       # (N,) transfer reason code

    @property
    def n_matchable(self) -> int:
        """Recall denominator: keypoints of A that a perfect matcher could match."""
        return int((self.partner >= 0).sum())


def ground_truth_partners(uv_a, uv_b, geom: PairGeometry, thresh_px: float = 3.0,
                          transfer: Transfer | None = None, **transfer_kw) -> Partners:
    """The recall denominator: which keypoints of A *have* a correct match in B.

    Precision alone can be maximised by matching almost nothing. Recall needs
    to know how many correct matches were available. A keypoint of A counts as
    matchable when:

    1. it has trustworthy depth,
    2. its point is in front of B and inside B's image,
    3. it is not occluded in B (and B has depth there to check), and
    4. B's detector actually fired within ``thresh_px`` of the predicted pixel.

    Condition 4 matters: a point visible in B but with no keypoint detected
    there is the *detector's* miss, not the matcher's. Without it, recall would
    mix detector repeatability into matcher quality.

    The denominator depends on ``thresh_px``, and at large thresholds it is
    inflated by chance: with ~1500 keypoints in a 640x480 image, the
    probability that *some* unrelated keypoint lies within r px of a given spot
    is about 1 - exp(-density * pi r^2) = 13% at 3 px and 32% at 5 px. So recall
    can *fall* as the threshold grows. Compare recall only at a fixed threshold.

    The partner is the nearest such keypoint. Several keypoints of B can lie
    within the threshold (detectors fire at multiple scales on one corner);
    :func:`label_matches` accepts any of them, so recall is consistent with it.
    """
    uv_a = np.asarray(uv_a, float).reshape(-1, 2)
    uv_b = np.asarray(uv_b, float).reshape(-1, 2)
    tr = transfer if transfer is not None else transfer_keypoints(uv_a, geom, **transfer_kw)

    n = len(uv_a)
    partner = np.full(n, -1)
    dist = np.full(n, np.nan)
    cand = np.nonzero(tr.reason == OK)[0]
    if geom.depth_b is None:
        # No occlusion check possible; visibility is assumed. Stated, not hidden.
        cand = np.nonzero(np.isin(tr.reason, (OK, NO_DEPTH_B)))[0]
    if len(cand) == 0 or len(uv_b) == 0:
        return Partners(partner, dist, tr.reason)

    # Brute-force nearest neighbour in chunks: ~2k x 2k keypoints is 4M
    # distances, trivial, and exact (no index structure to get wrong).
    for s in range(0, len(cand), 1024):
        c = cand[s:s + 1024]
        d = np.linalg.norm(tr.uv_b[c, None, :] - uv_b[None, :, :], axis=2)
        k = np.argmin(d, axis=1)
        dk = d[np.arange(len(c)), k]
        hit = dk <= thresh_px
        partner[c[hit]] = k[hit]
        dist[c[hit]] = dk[hit]
    return Partners(partner, dist, tr.reason)


def evaluate_matches(uv_a, uv_b, matches, geom: PairGeometry, thresh_px: float = 3.0,
                     **transfer_kw) -> dict:
    """Precision, recall and the counts behind them, for one image pair.

    * precision = correct / (correct + wrong)        -- UNKNOWN excluded
    * recall    = correct matches from matchable keypoints / matchable keypoints

    Recall's numerator only counts correct matches whose A-keypoint is
    matchable, so it can't exceed 1. A correct match from a keypoint that is
    occluded in B is counted in precision but not recall.
    """
    tr = transfer_keypoints(np.asarray(uv_a, float).reshape(-1, 2), geom, **transfer_kw)
    lab = label_matches(uv_a, uv_b, matches, geom, thresh_px, transfer=tr)
    par = ground_truth_partners(uv_a, uv_b, geom, thresh_px, transfer=tr)

    m = np.asarray(matches, dtype=int).reshape(-1, 2)
    hit = (lab.labels == CORRECT) & (par.partner[m[:, 0]] >= 0) if len(m) else np.zeros(0, bool)
    n_match = par.n_matchable
    reasons, counts = np.unique(tr.reason, return_counts=True)
    return {
        "precision": lab.precision,
        "recall": float(hit.sum() / n_match) if n_match else float("nan"),
        "n_matches": int(len(m)),
        "n_correct": lab.n_correct,
        "n_wrong": lab.n_wrong,
        "n_unknown": lab.n_unknown,
        "unknown_frac": lab.n_unknown / len(m) if len(m) else float("nan"),
        "n_matchable": n_match,
        "keypoints_a": int(len(tr.reason)),
        "transfer_reasons": {REASON_NAMES[int(r)]: int(c) for r, c in zip(reasons, counts)},
        "labels": lab,
        "partners": par,
    }


def pair_from_tum(dataset, i: int, j: int, interpolate: bool = True,
                  with_depth_b: bool = True, gt_time_offset: float = 0.0) -> PairGeometry:
    """PairGeometry for frames ``i`` (A) and ``j`` (B) of a :class:`TUMDataset`.

    ``T_camB_camA = T_world_camB^-1 * T_world_camA``. Read right to left: take
    a point in A's frame, lift it to the world with A's pose, then express it
    in B's frame with the inverse of B's pose.

    ``interpolate=True`` evaluates ground truth at each colour image's exact
    timestamp instead of using the nearest mocap sample. Mocap runs at 100 Hz,
    so the nearest sample can be up to ~5 ms off, which is ~3 mm of camera
    motion at fr1/xyz's 0.6 m/s peak, or ~1.5 px for a point 1 m away. That is
    half a typical 3 px threshold, spent before the matcher does anything.
    """
    from ..data.trajectory import interpolate_pose

    def pose(k):
        f = dataset.frames[k]
        if interpolate and dataset.groundtruth is not None:
            T = interpolate_pose(dataset.groundtruth, f.timestamp + gt_time_offset)
            if T is not None:
                return T
        if f.T_world_cam is None:
            raise ValueError(f"frame {k} has no ground-truth pose")
        return f.T_world_cam

    Ta, Tb = pose(i), pose(j)
    T_world_camA = SE3(Ta.R, Ta.t, "world", "camA")
    T_world_camB = SE3(Tb.R, Tb.t, "world", "camB")
    T_camB_camA = T_world_camB.inverse() @ T_world_camA
    return PairGeometry(
        camera=dataset.camera, T_camB_camA=T_camB_camA,
        depth_a=dataset.load_depth(i),
        depth_b=dataset.load_depth(j) if with_depth_b else None)
