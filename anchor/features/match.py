"""2D-2D descriptor matching.

Contract of the module's output: an (M, 2) int array of index pairs
``(i into Fa, j into Fb)``. Each ``i`` appears at most once (the labeller in
``anchor.eval.correspondence`` rejects duplicates). Whether a ``j`` may appear
more than once depends on ``mutual`` -- with ``mutual=False`` the same j can
appear; with ``mutual=True`` the result is one-to-one.

A match produced here is a *hypothesis*. Whether it is correct is a geometric
question (does it agree with the true pose?), answered by the labeller now and
by RANSAC in Step 4. Don't over-tighten here to compensate for a weak Step 4.
"""

from __future__ import annotations

import numpy as np

from .extract import Features


def pairwise_sq_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Squared L2 distance between every row of ``desc_a`` (N, D) and ``desc_b`` (M, D).

    Returns (N, M) float64. Uses the expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2*<a,b>
    to avoid the (N, M, D) intermediate. Result is clipped to [0, inf) to prevent
    negative values from floating-point cancellation.
    """
    a = np.asarray(desc_a, dtype=np.float64)
    b = np.asarray(desc_b, dtype=np.float64)
    a2 = (a * a).sum(axis=1)          # (N,)
    b2 = (b * b).sum(axis=1)          # (M,)
    D2 = a2[:, None] + b2[None, :] - 2.0 * (a @ b.T)
    return np.maximum(D2, 0.0)


def two_nearest(D2: np.ndarray):
    """Nearest and second-nearest neighbour of each row of a distance matrix.

    ``D2`` is (N, M) squared distances. Returns ``(nn_idx, d1, d2)``:

    * ``nn_idx`` (N,) int: column of the nearest neighbour.
    * ``d1`` (N,) float: **unsquared** distance to it.
    * ``d2`` (N,) float: **unsquared** distance to the second-nearest.

    When M == 0 or N == 0: nn_idx is empty, d1 and d2 are empty NaN arrays.
    When M == 1: d2 is NaN (no second neighbour exists). ``ratio_test`` treats
    NaN as a failed test, so single-candidate keypoints are always rejected.
    """
    n, m = D2.shape
    if n == 0 or m == 0:
        return np.empty(n, dtype=int), np.full(n, np.nan), np.full(n, np.nan)
    if m == 1:
        return np.zeros(n, dtype=int), np.sqrt(D2[:, 0]), np.full(n, np.nan)

    # argpartition with kth=1: element at position 1 is the second-smallest value.
    # The single element before it (position 0) must be the smallest. So the
    # first two columns are [nearest-index, second-nearest-index] in order.
    idx2 = np.argpartition(D2, kth=1, axis=1)[:, :2]   # (N, 2)
    sq = D2[np.arange(n)[:, None], idx2]                # (N, 2)
    return idx2[:, 0], np.sqrt(sq[:, 0]), np.sqrt(sq[:, 1])


def ratio_test(d1: np.ndarray, d2: np.ndarray, ratio: float) -> np.ndarray:
    """Lowe's ratio test. Returns a (N,) bool mask of rows that pass.

    Pass when ``d1 < ratio * d2`` on **unsquared** distances.

    When ``d2 == 0`` (duplicate descriptors in B, so d1 == d2 == 0):
    ``0 < ratio * 0`` is False, so the keypoint is rejected. No NaN is produced.

    When ``d2`` is NaN (M == 1, no second neighbour): the comparison returns
    False, rejecting the keypoint. No warning is raised.
    """
    with np.errstate(invalid="ignore"):   # silence NaN comparison warning
        return d1 < ratio * d2


def mutual_nn(nn_ab: np.ndarray, nn_ba: np.ndarray) -> np.ndarray:
    """Cross-check. Returns a (N,) bool mask over A's keypoints.

    ``nn_ab[i]`` is i's nearest neighbour in B; ``nn_ba[j]`` is j's nearest
    neighbour in A. Keep i only if ``nn_ba[nn_ab[i]] == i``. The result is
    one-to-one by construction (each j in B can be the nearest of at most one
    i that survives, because if two i's pointed to the same j, only one could
    be j's nearest neighbour in A).
    """
    if len(nn_ab) == 0:
        return np.zeros(0, dtype=bool)
    return nn_ba[nn_ab] == np.arange(len(nn_ab))


def match_2d2d(Fa: Features, Fb: Features, ratio: float | None = 0.8,
               mutual: bool = False, check_kind: bool = True) -> np.ndarray:
    """Match keypoints of image A to image B by descriptor.

    Parameters
    ----------
    ratio
        Lowe ratio threshold on unsquared distances, or ``None`` to disable.
    mutual
        Apply :func:`mutual_nn` as well. With mutual=True the result is
        one-to-one (each j appears at most once). With mutual=False a j can
        appear more than once.
    check_kind
        Refuse to compare descriptors of different kinds (e.g. SIFT vs
        RootSIFT). Mixing them still produces numbers but they are not
        comparable on the same scale.

    Returns
    -------
    (M, 2) int array of ``(i, j)``, sorted by ``i``. Empty input -> shape (0, 2).

    Raises
    ------
    ValueError
        If check_kind is True and the descriptor kinds differ.
    NotImplementedError
        If either feature set uses ORB (Hamming distance); a separate matcher
        is needed for binary descriptors.
    """
    if check_kind and Fa.kind != Fb.kind:
        raise ValueError(
            f"Descriptor kind mismatch: {Fa.kind!r} vs {Fb.kind!r}. "
            "Pass check_kind=False to override."
        )
    if Fa.kind == "orb" or Fb.kind == "orb":
        raise NotImplementedError(
            "ORB uses Hamming distance; L2-based matching is not appropriate. "
            "Implement a Hamming matcher separately."
        )

    if len(Fa) == 0 or len(Fb) == 0:
        return np.empty((0, 2), dtype=int)

    D2 = pairwise_sq_distances(Fa.desc, Fb.desc)   # (N, M)
    nn_ab, d1, d2 = two_nearest(D2)

    mask = np.ones(len(Fa), dtype=bool)

    if ratio is not None:
        mask &= ratio_test(d1, d2, ratio)

    if mutual:
        # D2.T is (M, N): row j is distances from B[j] to all A keypoints.
        nn_ba, _, _ = two_nearest(D2.T)
        mask &= mutual_nn(nn_ab, nn_ba)

    i_valid = np.nonzero(mask)[0]
    if len(i_valid) == 0:
        return np.empty((0, 2), dtype=int)

    # np.nonzero returns sorted indices, so output is sorted by i.
    return np.column_stack([i_valid, nn_ab[i_valid]])
