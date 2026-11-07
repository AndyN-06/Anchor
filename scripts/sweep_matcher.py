#!/usr/bin/env python3
"""Precision/recall of match_2d2d vs ratio threshold on real TUM pairs.

The Step 3 review deliverable (ARCHITECTURE.md 4.5). For each configuration
in {ratio only, mutual only, ratio + mutual} x {SIFT, RootSIFT} and ratio in
0.60..0.95, score the hand-written matcher with the labeller
(anchor.eval.correspondence) on slow pairs (both frames < 15 deg/s, gaps of
5 and 15 frames, 3 px threshold), and the OpenCV oracle on the same pairs.

Reference to beat: oracle SIFT, ratio 0.8, gap 5, slow pairs:
precision 0.82, recall 0.62 at 3 px (scripts/label_tum_pair.py).

    python scripts/sweep_matcher.py data/rgbd_dataset_freiburg1_xyz --figure pr.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def select_pairs(dataset, gaps, n_per_gap, max_deg_s, seed=0):
    """Frame pairs (i, i+gap) with both frames under ``max_deg_s``. Fixed seed."""
    raise NotImplementedError


def score_config(dataset, pairs, ratio, mutual, root_sift, thresh_px):
    """Pooled precision/recall over ``pairs`` for one matcher configuration."""
    raise NotImplementedError


def plot_pr(results, path):
    """One curve per configuration, oracle as a reference point."""
    raise NotImplementedError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--figure", type=Path, default=None)
    args = ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
