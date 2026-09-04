#!/usr/bin/env python3
"""Sanity-check a TUM RGB-D sequence and exercise the evaluation harness.

Run this on a freshly downloaded sequence before trusting anything built on
top of it. It answers, from the data rather than from documentation:

* Did all three streams (colour, depth, ground truth) associate cleanly?
* Is the ground-truth pose ``T_world_cam`` or its inverse?
* Are the intrinsics, depth scale and unprojection mutually consistent?

The last two are the same measurement: reproject one frame's depth into
another and compare against what that frame measured. If the intrinsics were
wrong, or the depth scale off, or the pose convention inverted, this residual
blows up. Sub-centimetre agreement means the whole chain is consistent.

Then it runs ATE/RPE end to end on a deliberately corrupted copy of ground
truth, which is the only honest way to exercise the metrics before a real
estimator exists.

    python scripts/check_tum_sequence.py data/rgbd_dataset_freiburg1_xyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor.data.tum import TUMDataset, verify_groundtruth_convention  # noqa: E402
from anchor.data.trajectory import Trajectory                          # noqa: E402
from anchor.eval.metrics import ate, rpe                               # noqa: E402
from anchor.eval.plots import evaluation_figure                        # noqa: E402
from anchor.geometry.se3 import SE3                                    # noqa: E402


def corrupt(traj: Trajectory, drift_per_s: float, noise_m: float, seed: int = 0):
    """Ground truth plus linear drift and per-frame noise -- a stand-in estimate."""
    rng = np.random.default_rng(seed)
    t0 = traj.timestamps[0]
    poses = []
    for t, T in zip(traj.timestamps, traj.poses):
        offset = np.array([drift_per_s * (t - t0), 0.0, 0.0])
        poses.append(SE3(T.R, T.t + offset + rng.normal(scale=noise_m, size=3),
                         "world", "cam"))
    return Trajectory(traj.timestamps.copy(), poses, "simulated estimate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--gap", type=int, default=30,
                    help="frame gap for the geometric consistency check")
    ap.add_argument("--drift", type=float, default=0.02, help="simulated drift, m/s")
    ap.add_argument("--noise", type=float, default=0.005, help="simulated noise, m")
    ap.add_argument("--figure", type=Path, default=None, help="write the report figure here")
    args = ap.parse_args()

    ds = TUMDataset(args.sequence)
    traj = ds.trajectory()

    print(f"sequence        {args.sequence.name}")
    print(f"camera          {ds.category}  fx={ds.camera.fx:.2f} cx={ds.camera.cx:.2f} "
          f"depth_scale={ds.depth_scale:g}")
    print(f"frames          {len(ds)} of {ds.n_rgb_total} colour images associated")
    print(f"duration        {traj.duration:.2f} s over {traj.path_length():.2f} m "
          f"({traj.path_length()/max(traj.duration,1e-9):.2f} m/s mean speed)")
    dt_depth = np.array([f.dt_depth for f in ds.frames])
    print(f"colour->depth   mean {dt_depth.mean()*1000:.1f} ms, max {dt_depth.max()*1000:.1f} ms")

    depth = ds.load_depth(0)
    print(f"depth frame 0   {100*np.isfinite(depth).mean():.1f}% valid, "
          f"{np.nanmin(depth):.2f}-{np.nanmax(depth):.2f} m")

    print("\n-- ground-truth frame convention (measured, not assumed) --")
    rep = verify_groundtruth_convention(ds, 0, args.gap)
    for key in ("as_T_world_cam", "as_T_cam_world"):
        r = rep[key]
        print(f"  {key:16s} median depth residual "
              f"{r['median_depth_residual_m']*100:7.2f} cm  "
              f"({r.get('n_compared', 0)} points)")
    print(f"  verdict: {rep['verdict']}  ({rep['ratio']:.1f}x better than the alternative)")
    if rep["ratio"] < 2.0:
        print("  WARNING: not decisive. Increase --gap so the camera moves further.")

    print("\n-- evaluation harness on a simulated estimate --")
    est = corrupt(traj, args.drift, args.noise)
    a = ate(est, traj, align="se3")
    r = rpe(est, traj, delta=1.0, delta_unit="s")
    print(a.summary())
    print(r.summary())
    print(f"  (injected: {args.drift*100:.1f} cm/s drift, {args.noise*100:.1f} cm noise)")

    if args.figure:
        evaluation_figure(est.transformed(a.alignment, a.scale), traj, a, r,
                          path=args.figure, suptitle=f"{args.sequence.name} — simulated estimate")
        print(f"\nfigure written to {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
