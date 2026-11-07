"""2D-2D descriptor matching. CARVE-OUT: implement by hand, no cv2 matchers.

Contract of the module's output: an (M, 2) int array of index pairs
``(i into Fa, j into Fb)``. Each ``i`` appears at most once (the labeller in
``anchor.eval.correspondence`` rejects duplicates). Whether a ``j`` may appear
more than once depends on ``mutual`` -- decide and document.

A match produced here is a *hypothesis*. Whether it is correct is a geometric
question (does it agree with the true pose?), answered by the labeller now and
by RANSAC in Step 4. Don't over-tighten here to compensate for a weak Step 4.
"""

from __future__ import annotations

import numpy as np

from .extract import Features


def pairwise_sq_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Squared L2 distance between every row of ``desc_a`` (N, D) and ``desc_b`` (M, D).

    Returns (N, M) float64. Must be computed without an (N, M, D) intermediate
    (2000 x 2000 x 128 float64 is 4 GB).

    Numerical requirement: the result is never negative, including for
    identical descriptors.
    """
    raise NotImplementedError


def two_nearest(D2: np.ndarray):
    """Nearest and second-nearest neighbour of each row of a distance matrix.

    ``D2`` is (N, M) squared distances. Returns ``(nn_idx, d1, d2)``:

    * ``nn_idx`` (N,) int: column of the nearest neighbour.
    * ``d1`` (N,) float: distance to it (**not squared** -- say which you return
      and keep it consistent with :func:`ratio_test`).
    * ``d2`` (N,) float: distance to the second-nearest.

    Decide the behaviour when M < 2 (no second neighbour) and when N or M is 0.
    """
    raise NotImplementedError


def ratio_test(d1: np.ndarray, d2: np.ndarray, ratio: float) -> np.ndarray:
    """Lowe's ratio test. Returns a (N,) bool mask of rows that pass.

    Pass when the nearest neighbour is clearly better than the runner-up:
    ``d1 < ratio * d2`` on (unsquared) distances.

    Decide what happens when ``d2 == 0`` and when ``d2`` is undefined.
    """
    raise NotImplementedError


def mutual_nn(nn_ab: np.ndarray, nn_ba: np.ndarray) -> np.ndarray:
    """Cross-check. Returns a (N,) bool mask over A's keypoints.

    ``nn_ab[i]`` is i's nearest neighbour in B; ``nn_ba[j]`` is j's nearest
    neighbour in A. Keep i only if ``nn_ba[nn_ab[i]] == i``. The result is
    one-to-one by construction.
    """
    raise NotImplementedError


def match_2d2d(Fa: Features, Fb: Features, ratio: float | None = 0.8,
               mutual: bool = False, check_kind: bool = True) -> np.ndarray:
    """Match keypoints of image A to image B by descriptor.

    Parameters
    ----------
    ratio
        Lowe ratio threshold on unsquared distances, or ``None`` to disable.
    mutual
        Apply :func:`mutual_nn` as well.
    check_kind
        Refuse to compare descriptors of different kinds (e.g. SIFT vs
        RootSIFT). Mixing them still produces numbers.

    Returns
    -------
    (M, 2) int array of ``(i, j)``, sorted by ``i``. Empty input -> shape (0, 2).
    """
    raise NotImplementedError
