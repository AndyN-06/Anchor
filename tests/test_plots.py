"""Smoke tests for the report figures.

These check that the plotting code runs and writes a file. They cannot check
that a figure is *readable* -- that needs a human looking at the PNG, which is
step 2's manual verification, not something pytest can assert.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")           # headless: no display in CI or in a sandbox

from anchor.eval.metrics import ate, rpe                      # noqa: E402
from anchor.eval.plots import evaluation_figure               # noqa: E402
from tests.test_metrics import circle_trajectory, drifted     # noqa: E402


@pytest.fixture
def results():
    gt = circle_trajectory(n=300)
    est = drifted(gt, drift_per_second=0.02)
    return est, gt, ate(est, gt, align="se3"), rpe(est, gt, delta=1.0)


def test_evaluation_figure_writes_a_file(tmp_path, results):
    est, gt, a, r = results
    path = tmp_path / "figs" / "eval.png"
    fig = evaluation_figure(est.transformed(a.alignment, a.scale), gt, a, r,
                            path=path, suptitle="synthetic drift")
    assert path.exists() and path.stat().st_size > 10_000
    assert len(fig.axes) == 5   # trajectory, summary text, error-over-time, 2 histograms
    matplotlib.pyplot.close(fig)


def test_evaluation_figure_without_rpe(tmp_path, results):
    est, gt, a, _ = results
    fig = evaluation_figure(est.transformed(a.alignment, a.scale), gt, a,
                            path=tmp_path / "eval2.png")
    assert (tmp_path / "eval2.png").exists()
    matplotlib.pyplot.close(fig)
