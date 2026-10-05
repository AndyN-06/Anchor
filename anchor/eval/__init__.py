"""Evaluation: trajectory alignment, ATE, RPE, plots, and ground-truth match labels."""

from .align import alignment_is_degenerate, umeyama_alignment
from .correspondence import (
    CORRECT, UNKNOWN, WRONG, PairGeometry, evaluate_matches, ground_truth_partners,
    label_matches, pair_from_tum, transfer_keypoints,
)
from .metrics import ATEResult, RPEResult, ate, rpe
from .plots import evaluation_figure, plot_error_over_time, plot_rpe_histogram, plot_trajectory_xy

__all__ = ["umeyama_alignment", "alignment_is_degenerate", "ate", "rpe",
           "evaluation_figure", "plot_trajectory_xy", "plot_error_over_time",
           "plot_rpe_histogram",
           "ATEResult", "RPEResult",
           "PairGeometry", "transfer_keypoints", "label_matches", "ground_truth_partners",
           "evaluate_matches", "pair_from_tum", "CORRECT", "WRONG", "UNKNOWN"]
