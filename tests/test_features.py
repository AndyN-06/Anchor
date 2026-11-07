"""Step 3: extraction and matching. Bodies are yours; each docstring is the spec."""

import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="Step 3 not implemented yet")


# ----------------------------------------------------------------- extraction
def test_extract_returns_consistent_shapes():
    """uv (N,2), xn (N,2), desc (N,128) float32, scale/angle/response (N,)."""


def test_extract_on_blank_image_returns_empty_arrays_not_none():
    """A uniform image has no keypoints. Shapes (0,2), (0,128), ... not None."""


def test_xn_matches_camera_pixel_to_normalized():
    """xn equals camera.pixel_to_normalized(uv) exactly (same function, same input)."""


def test_rootsift_rows_have_unit_norm_and_hellinger_distance():
    """||r(a)||=1, and ||r(a)-r(b)||^2 == 2 - 2*sum(sqrt(a_k b_k)/sqrt(|a|_1 |b|_1))."""


# -------------------------------------------------------------- distances
def test_pairwise_sq_distances_matches_brute_force():
    """Against np.sum((a[:,None]-b[None])**2, -1) on small random inputs, ~1e-9 rel."""


def test_pairwise_sq_distances_never_negative_for_identical_rows():
    """desc_b = desc_a (SIFT-scale values, ~0-255): diagonal is >= 0, and ~0."""


def test_two_nearest_edge_cases():
    """M == 1 (no second neighbour), N == 0, M == 0: documented behaviour, no crash."""


# --------------------------------------------------------------- filtering
def test_ratio_test_uses_unsquared_distances():
    """d1=0.85, d2=1.0: fails at ratio 0.8. (On squared distances it would pass: 0.72 < 0.8.)"""


def test_ratio_test_with_zero_second_distance():
    """d1 = d2 = 0 (duplicate descriptors in B): documented, no NaN, no warning."""


def test_mutual_nn_is_one_to_one():
    """No j appears twice in the output."""


def test_match_2d2d_never_repeats_an_a_index():
    """Required by the labeller; holds for every (ratio, mutual) combination."""


def test_match_2d2d_rejects_mixed_descriptor_kinds():
    """SIFT vs RootSIFT features -> ValueError when check_kind=True."""


# ---------------------------------------------------------------- synthetic
def test_synthetic_precision_recall_is_exact():
    """Random 128-D descriptor per 3D point (Step 1 scene); each view sees it plus
    Gaussian noise, plus distractor keypoints with fresh descriptors. Every true
    pair is known, so precision/recall are exact. Assert recall falls and
    precision rises as ratio goes 0.95 -> 0.6."""


# ------------------------------------------------------------------- oracle
def test_ratio_only_reproduces_opencv_bf_knn():
    """mutual=False, same ratio: identical (i, j) set to cv2.BFMatcher(NORM_L2)
    .knnMatch(k=2) + ratio test on the same descriptors. Ties are the only
    permitted difference -- construct inputs without exact ties."""


def test_mutual_reproduces_opencv_crosscheck():
    """ratio=None, mutual=True: identical to cv2.BFMatcher(NORM_L2, crossCheck=True).match."""
