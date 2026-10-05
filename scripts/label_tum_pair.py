#!/usr/bin/env python3
"""Run the ground-truth match labeller on real TUM frame pairs.

Two jobs:

1. **Measure the labeller's own error floor** before trusting it. Even a
   perfect match shows a nonzero error here: Kinect depth noise, mocap pose
   error, and the colour-depth timing offset all move the predicted pixel. The
   error histogram of a decent matcher is bimodal -- a tight peak (correct
   matches, width = the floor) and a long flat tail (wrong matches). The
   threshold belongs in the gap between them, not at a number picked in
   advance.

2. **Give a reference curve** for the hand-written matcher in Step 3: the same
   pairs, scored by the same labeller, so the two are directly comparable.

The matcher here is OpenCV's brute-force kNN + Lowe ratio test. It is an
ORACLE, used only to exercise the labeller and as a baseline. It must not be
imported by the pipeline (ARCHITECTURE.md section 1.1).

    python scripts/label_tum_pair.py data/rgbd_dataset_freiburg1_xyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from anchor.data.trajectory import camera_speed                    # noqa: E402
from anchor.data.tum import TUMDataset                              # noqa: E402
from anchor.eval.correspondence import (                            # noqa: E402
    evaluate_matches, label_matches, pair_from_tum, transfer_keypoints,
)


def oracle_sift_matches(img_a, img_b, ratio, n_features):
    """ORACLE ONLY. SIFT + brute-force L2 kNN + ratio test, via OpenCV."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    ka, da = sift.detectAndCompute(img_a, None)
    kb, db = sift.detectAndCompute(img_b, None)
    uv_a = np.array([k.pt for k in ka], float)
    uv_b = np.array([k.pt for k in kb], float)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    m = [(p[0].queryIdx, p[0].trainIdx) for p in knn
         if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return uv_a, uv_b, np.array(m, int).reshape(-1, 2)


def frame_speeds(ds):
    """Angular speed (deg/s) at every frame, from interpolated ground truth."""
    out = np.full(len(ds), np.inf)
    for k, f in enumerate(ds.frames):
        v = camera_speed(ds.groundtruth, f.timestamp)
        if v is not None:
            out[k] = np.degrees(v[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--gaps", type=int, nargs="+", default=[5, 15, 30, 60])
    ap.add_argument("--pairs", type=int, default=8, help="pairs per gap and speed bucket")
    ap.add_argument("--slow", type=float, default=15.0,
                    help="max angular speed (deg/s) at both frames for the 'slow' set")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--features", type=int, default=2000)
    args = ap.parse_args()

    ds = TUMDataset(args.sequence)
    w = frame_speeds(ds)
    print(f"{args.sequence.name}: oracle = SIFT + BF kNN, ratio {args.ratio}. "
          f"Angular speed per frame: median {np.median(w[np.isfinite(w)]):.1f} deg/s\n")

    rng = np.random.default_rng(0)
    for gap in args.gaps:
        starts = np.arange(len(ds) - gap)
        both = np.maximum(w[starts], w[starts + gap])
        for name, sel in (("slow", both < args.slow), ("fast", both >= 2 * args.slow)):
            cand = starts[sel]
            if len(cand) == 0:
                print(f"-- gap {gap}, {name}: no pairs\n")
                continue
            chosen = rng.choice(cand, size=min(args.pairs, len(cand)), replace=False)

            errs, offsets, per_th = [], [], {th: [0, 0, 0, 0] for th in (1.0, 2.0, 3.0, 5.0)}
            for i in sorted(chosen):
                j = int(i + gap)
                uv_a, uv_b, m = oracle_sift_matches(ds.load_gray(int(i)), ds.load_gray(j),
                                                    args.ratio, args.features)
                if len(m) == 0:
                    continue
                geom = pair_from_tum(ds, int(i), j)
                tr = transfer_keypoints(uv_a, geom)
                e = label_matches(uv_a, uv_b, m, geom, thresh_px=1e9, transfer=tr).error_px
                errs.append(e[np.isfinite(e)])
                r = uv_b[m[:, 1]] - tr.uv_b[m[:, 0]]
                near = np.isfinite(r).all(axis=1) & (np.linalg.norm(r, axis=1) < 8)
                if near.sum() >= 20:
                    offsets.append(np.linalg.norm(np.median(r[near], axis=0)))
                for th, acc in per_th.items():
                    res = evaluate_matches(uv_a, uv_b, m, geom, thresh_px=th)
                    hit = int(round(res["recall"] * res["n_matchable"])) if res["n_matchable"] else 0
                    acc[0] += res["n_correct"]; acc[1] += res["n_wrong"]
                    acc[2] += hit; acc[3] += res["n_matchable"]

            e = np.concatenate(errs)
            q = np.percentile(e, [25, 50, 75, 90])
            print(f"-- gap {gap} frames ({gap/30:.2f} s), {name} (<{args.slow:g} deg/s)"
                  if name == "slow" else
                  f"-- gap {gap} frames ({gap/30:.2f} s), {name} (>={2*args.slow:g} deg/s)", end="")
            print(f": {len(errs)} pairs, {len(e)} labelled matches --")
            print(f"   error px: p25 {q[0]:.2f}  p50 {q[1]:.2f}  p75 {q[2]:.2f}  p90 {q[3]:.1f}   "
                  f"per-pair systematic offset: median {np.median(offsets):.2f}, "
                  f"max {np.max(offsets):.2f} px")
            hist, edges = np.histogram(e, bins=[0, 0.5, 1, 1.5, 2, 3, 4, 5, 8, 15, 50, np.inf])
            print("   histogram  " + "  ".join(f"<{b:g}:{h}" for b, h in zip(edges[1:], hist)))
            for th, (c, wr, hit, n) in per_th.items():
                print(f"   thresh {th:3.0f} px: precision {c/max(c+wr,1):.3f}  recall {hit/max(n,1):.3f}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
