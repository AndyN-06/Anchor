"""Step 3: feature extraction and 2D-2D matching.

Detection and description may call OpenCV. Matching -- distances, nearest
neighbours, and every decision about which matches survive -- is hand-written
(carve-out, ARCHITECTURE.md section 4). OpenCV's matchers appear only in tests
and scripts, as oracles.
"""

from .extract import Features, detect_and_describe, extract, rootsift
from .match import match_2d2d, mutual_nn, pairwise_sq_distances, ratio_test, two_nearest

__all__ = [
    "Features", "detect_and_describe", "extract", "rootsift",
    "pairwise_sq_distances", "two_nearest", "ratio_test", "mutual_nn", "match_2d2d",
]
