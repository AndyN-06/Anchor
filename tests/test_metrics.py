"""ATE and RPE: the invariances that define them, and where they disagree."""

import numpy as np
import pytest

from anchor.data.trajectory import Trajectory
from anchor.eval.metrics import ate, rpe
from anchor.geometry.rotations import exp_so3
from anchor.geometry.se3 import SE3

RNG = np.random.default_rng(31)


def circle_trajectory(n=200, dt=1 / 30, radius=1.0, z_rate=0.0):
    """A camera on a circle, yawing as it goes.

    ``z_rate`` adds a steady climb. It defaults to zero, and that default is
    load-bearing rather than cosmetic -- see
    ``test_alignment_can_absorb_drift_that_matches_the_trajectory_shape``.
    """
    ts = np.arange(n) * dt
    poses = []
    for i in range(n):
        a = 0.03 * i
        R = exp_so3(np.array([0.1, -0.05, a]))
        poses.append(SE3(R, np.array([radius * np.cos(a), radius * np.sin(a), z_rate * i]),
                         "world", "cam"))
    return Trajectory(ts, poses, "gt")


def drifted(traj, drift_per_second=0.01):
    """Ground truth plus a slow linear position drift -- the classic SLAM failure."""
    t0 = traj.timestamps[0]
    poses = [SE3(T.R, T.t + np.array([drift_per_second * (t - t0), 0.0, 0.0]), "world", "cam")
             for t, T in zip(traj.timestamps, traj.poses)]
    return Trajectory(traj.timestamps.copy(), poses, "est")


def jittered(traj, sigma=0.005, seed=0):
    """Ground truth plus zero-mean per-frame noise -- no drift, bad local accuracy."""
    rng = np.random.default_rng(seed)
    poses = [SE3(T.R, T.t + rng.normal(scale=sigma, size=3), "world", "cam")
             for T in traj.poses]
    return Trajectory(traj.timestamps.copy(), poses, "est")


# ------------------------------------------------------------------------ ATE

def test_ate_of_a_trajectory_against_itself_is_zero():
    gt = circle_trajectory()
    for mode in ("none", "se3", "sim3"):
        r = ate(gt, gt, align=mode)
        assert r.translation["rmse"] < 1e-12
        assert r.rotation_deg["rmse"] < 1e-9
        assert r.n_pairs == len(gt)


def test_ate_is_invariant_to_a_global_rigid_transform():
    """This is the definition of ATE: the estimate's coordinate frame is
    arbitrary, so moving the whole trajectory must not change the score."""
    gt = circle_trajectory()
    T = SE3(exp_so3(np.array([0.9, -0.4, 0.2])), np.array([5.0, -3.0, 2.0]))
    moved = gt.transformed(T)
    assert ate(moved, gt, align="se3").translation["rmse"] < 1e-9
    # ...and without alignment the same trajectory scores terribly, which is
    # why comparing raw frames is meaningless.
    assert ate(moved, gt, align="none").translation["rmse"] > 1.0


def test_se3_alignment_does_not_hide_a_scale_error():
    """A system whose distances are all 10% too long is broken. SE(3) must
    report that; Sim(3) will absorb it, which is why the alignment mode has to
    be reported alongside the number."""
    gt = circle_trajectory()
    scaled = gt.transformed(SE3.identity(), scale=1.10)
    assert ate(scaled, gt, align="se3").translation["rmse"] > 0.02
    assert ate(scaled, gt, align="sim3").translation["rmse"] < 1e-9
    assert ate(scaled, gt, align="sim3").scale == pytest.approx(1 / 1.10, rel=1e-6)


def test_ate_grows_with_drift():
    gt = circle_trajectory()
    small = ate(drifted(gt, 0.005), gt, align="se3").translation["rmse"]
    large = ate(drifted(gt, 0.05), gt, align="se3").translation["rmse"]
    assert large > small > 0
    assert large / small == pytest.approx(10.0, rel=0.05)   # linear in drift rate


def test_ate_reports_rotation_error_too():
    gt = circle_trajectory()
    tilted = Trajectory(gt.timestamps.copy(),
                        [SE3(T.R @ exp_so3(np.array([0.0, 0.0, np.radians(2.0)])), T.t,
                             "world", "cam") for T in gt.poses])
    r = ate(tilted, gt, align="none")
    assert r.rotation_deg["rmse"] == pytest.approx(2.0, rel=1e-6)
    assert r.translation["rmse"] < 1e-12


def test_ate_rejects_unknown_alignment_mode():
    gt = circle_trajectory()
    with pytest.raises(ValueError, match="align must be"):
        ate(gt, gt, align="procrustes")


def test_ate_fails_loudly_when_trajectories_do_not_overlap_in_time():
    gt = circle_trajectory()
    shifted = Trajectory(gt.timestamps + 1000.0, gt.poses)
    with pytest.raises(ValueError, match="do not overlap in time"):
        ate(shifted, gt)


def test_stats_expose_the_shape_of_the_error_not_just_rmse():
    """RMSE alone cannot distinguish 'uniformly mediocre' from 'perfect except
    for one catastrophic failure'. Median and max must."""
    gt = circle_trajectory(n=100)
    est_positions = gt.positions.copy()
    est_positions[50] += np.array([1.0, 0.0, 0.0])          # one bad frame
    spike = Trajectory(gt.timestamps.copy(),
                       [SE3(T.R, p, "world", "cam") for T, p in zip(gt.poses, est_positions)])
    r = ate(spike, gt, align="none")
    assert r.translation["median"] < 1e-9
    assert r.translation["max"] == pytest.approx(1.0)
    assert r.translation["rmse"] > 0.09


# ------------------------------------------------------------------------ RPE

def test_rpe_of_a_trajectory_against_itself_is_zero():
    gt = circle_trajectory()
    r = rpe(gt, gt, delta=1.0, delta_unit="s")
    assert r.translation["rmse"] < 1e-12
    assert r.rotation_deg["rmse"] < 1e-9
    assert r.n_pairs > 100


def test_rpe_needs_no_alignment():
    """RPE compares relative motions, so both world frames cancel algebraically.
    Moving the estimate anywhere must leave the number untouched."""
    gt = circle_trajectory()
    T = SE3(exp_so3(np.array([1.2, 0.3, -0.8])), np.array([10.0, 4.0, -7.0]))
    r = rpe(gt.transformed(T), gt, delta=1.0)
    assert r.translation["rmse"] < 1e-9
    assert r.rotation_deg["rmse"] < 1e-9


def test_rpe_and_ate_disagree_on_purpose():
    """The reason both are reported.

    Slow smooth drift: large ATE (the trajectory ends up far from truth) but
    small RPE (each one-second interval is almost exactly right).
    Zero-mean jitter: small ATE (it never goes anywhere) but large RPE (every
    short interval is wrong).

    A system reporting only one of these can hide its actual failure mode.

    Parameters come from the algebra so the test is not knife-edge. For a
    linear drift of rate d over duration T, alignment removes the mean and the
    post-alignment ATE RMSE is d*T/sqrt(12); the 1-second RPE is just d. For
    isotropic jitter of per-axis sigma, ATE is sqrt(3)*sigma and RPE is
    sqrt(6)*sigma -- differencing two independent samples doubles the variance.
    Separating both directions by a factor of 2 therefore needs

        2*sqrt(3)*sigma*sqrt(12)/T  <  d  <  sqrt(6)*sigma/2

    With T = 20 s and sigma = 2.4 cm that window is 1.4 cm/s < d < 2.9 cm/s,
    and d = 2 cm/s sits near its middle.
    """
    gt = circle_trajectory(n=600)          # 20 s, so drift has room to accumulate
    assert gt.duration == pytest.approx(20.0, rel=0.01)

    d = drifted(gt, drift_per_second=0.02)
    ate_drift = ate(d, gt, align="se3").translation["rmse"]
    rpe_drift = rpe(d, gt, delta=1.0).translation["rmse"]

    j = jittered(gt, sigma=0.024)
    ate_jitter = ate(j, gt, align="se3").translation["rmse"]
    rpe_jitter = rpe(j, gt, delta=1.0).translation["rmse"]

    assert ate_drift > 2 * ate_jitter      # drift wrecks global consistency
    assert rpe_jitter > 2 * rpe_drift      # jitter wrecks local accuracy


def test_alignment_can_absorb_drift_that_matches_the_trajectory_shape():
    """A caveat on ATE that cost an hour to find, so it lives in the suite.

    ATE aligns the estimate to ground truth before measuring, and alignment has
    6 free parameters. If the drift happens to look like something a rigid
    transform can produce, alignment eats it and ATE under-reports.

    Concretely: give ground truth a steady climb in z, so its position has a
    component linear in time. Now add a drift along x that is also linear in
    time. A small rotation mixing z into x produces exactly that displacement,
    so the aligner applies one and the drift largely disappears -- measured
    here, roughly 80% of it. On a trajectory with no linear-in-time component,
    99% of the same drift survives.

    Consequences for reporting: an ATE number is a statement about the estimate
    *and* the trajectory shape it was measured on. It is not a lower bound on
    drift, and it is the reason RPE gets reported next to it -- RPE needs no
    alignment, so it has nothing to absorb the drift with, and it returned the
    true 2 cm/s in both cases.
    """
    flat = circle_trajectory(n=600, z_rate=0.0)
    ramped = circle_trajectory(n=600, z_rate=0.02)       # climbs steadily

    ate_flat = ate(drifted(flat, 0.02), flat, align="se3").translation["rmse"]
    ate_ramped = ate(drifted(ramped, 0.02), ramped, align="se3").translation["rmse"]

    expected = 0.02 * flat.duration / np.sqrt(12)        # d*T/sqrt(12)
    assert ate_flat == pytest.approx(expected, rel=0.05)  # drift survives alignment
    assert ate_ramped < 0.3 * ate_flat                    # ...and here it does not

    # RPE is unfooled in both cases, because it never aligns anything.
    assert rpe(drifted(flat, 0.02), flat, delta=1.0).translation["rmse"] == pytest.approx(0.02, rel=0.02)
    assert rpe(drifted(ramped, 0.02), ramped, delta=1.0).translation["rmse"] == pytest.approx(0.02, rel=0.02)


def test_rpe_translation_equals_the_known_drift_rate():
    """A drift of 5 cm/s must show up as ~5 cm per 1-second interval."""
    gt = circle_trajectory(n=400)
    r = rpe(drifted(gt, drift_per_second=0.05), gt, delta=1.0, delta_unit="s")
    assert r.translation["median"] == pytest.approx(0.05, rel=0.05)


def test_rpe_frame_and_second_deltas_agree_at_the_frame_rate():
    gt = circle_trajectory(n=300, dt=1 / 30)
    est = drifted(gt, 0.05)
    by_sec = rpe(est, gt, delta=1.0, delta_unit="s").translation["median"]
    by_frame = rpe(est, gt, delta=30, delta_unit="f").translation["median"]
    assert by_sec == pytest.approx(by_frame, rel=0.05)


def test_rpe_scale_argument_corrects_a_monocular_estimate():
    """Without the scale from Sim(3) alignment, a monocular trajectory reports
    its unknown scale factor as if it were motion error."""
    gt = circle_trajectory()
    est = gt.transformed(SE3.identity(), scale=0.5)
    uncorrected = rpe(est, gt, delta=1.0).translation["rmse"]
    corrected = rpe(est, gt, delta=1.0, scale=2.0).translation["rmse"]
    assert uncorrected > 0.01
    assert corrected < 1e-9


def test_rpe_rejects_bad_arguments():
    gt = circle_trajectory()
    with pytest.raises(ValueError, match="delta_unit"):
        rpe(gt, gt, delta_unit="minutes")
    with pytest.raises(ValueError, match="no pose pairs"):
        rpe(gt, gt, delta=1000.0, delta_unit="s")


def test_ate_rotation_error_is_contaminated_by_the_alignment():
    """A trap in ATE that produces confidently wrong conclusions.

    The alignment is fitted to positions only, then applied to orientations
    too. So when the aligner rotates the trajectory to better fit a
    translation drift, it manufactures an orientation offset out of nothing.

    Here the estimate has *exactly* ground truth's rotations -- zero rotation
    error by construction -- yet ATE reports a large one, equal to the
    alignment's own rotation. RPE, which never aligns, correctly reports zero.
    """
    gt = circle_trajectory(n=600)
    est = drifted(gt, drift_per_second=0.05)          # translation only
    for T_est, T_gt in zip(est.poses, gt.poses):
        assert np.allclose(T_est.R, T_gt.R)           # no rotation error injected

    a = ate(est, gt, align="se3")
    assert a.alignment_rotation_deg > 1.0
    assert a.rotation_deg["rmse"] == pytest.approx(a.alignment_rotation_deg, rel=1e-6)
    assert "Use RPE rotation instead" in a.summary()

    assert rpe(est, gt, delta=1.0).rotation_deg["rmse"] < 1e-9
