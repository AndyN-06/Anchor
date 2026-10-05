"""Ground-truth match labelling: closed-form scenes where the answer is known."""

import numpy as np
import pytest

from anchor.data.trajectory import Trajectory, interpolate_pose
from anchor.eval.correspondence import (
    BEHIND_B, CORRECT, DEPTH_EDGE_A, NO_DEPTH_A, OCCLUDED_B, OK, OUT_OF_VIEW_B, UNKNOWN,
    WRONG, PairGeometry, evaluate_matches, ground_truth_partners, label_matches,
    transfer_keypoints,
)
from anchor.geometry.camera import PinholeCamera
from anchor.geometry.rotations import exp_so3
from anchor.geometry.se3 import SE3

IDEAL = PinholeCamera(500.0, 500.0, 319.5, 239.5, 640, 480)
FR1 = PinholeCamera(517.306408, 516.469215, 318.643040, 255.313989, 640, 480,
                    [0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
Z = 2.0          # fronto-parallel wall, metres
BASELINE = 0.1   # camera B is 10 cm to the right of A


def wall(cam, z=Z):
    return np.full((cam.height, cam.width), z, dtype=float)


def stereo_pair(cam=IDEAL, depth_b=True):
    """B = A moved +10 cm along A's x axis, same orientation.

    Camera B's centre in A is (0.1, 0, 0), so T_camA_camB is a translation of
    +0.1 and T_camB_camA (what the labeller needs) is a translation of -0.1:
    a point at x in A sits at x - 0.1 in B. The camera moved right, so the
    scene shifts left in the image by fx * 0.1 / z pixels -- 25 px here.
    """
    T_camB_camA = SE3(np.eye(3), [-BASELINE, 0.0, 0.0], "camB", "camA")
    return PairGeometry(cam, T_camB_camA, wall(cam), wall(cam) if depth_b else None)


def grid(n=10, margin=60):
    us = np.linspace(margin, 640 - margin, n)
    vs = np.linspace(margin, 480 - margin, n)
    return np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2)


def test_transfer_matches_the_closed_form_disparity():
    uv_a = grid()
    tr = transfer_keypoints(uv_a, stereo_pair())
    assert (tr.reason[uv_a[:, 0] > 30] == OK).all()
    expected = uv_a - np.array([IDEAL.fx * BASELINE / Z, 0.0])     # 25 px left
    assert np.allclose(tr.uv_b, expected, atol=1e-9)
    assert np.allclose(tr.z_b, Z)


def test_transfer_with_real_distortion_round_trips():
    """With fr1 distortion there's no closed form for the shift, so check
    self-consistency: lift the transferred pixel with B's pose and it must be
    the same 3D point. This exercises undistort (lift) and distort (project)
    against each other."""
    geom = stereo_pair(FR1)
    uv_a = grid(margin=80)
    tr = transfer_keypoints(uv_a, geom)
    ok = tr.reason == OK
    P_b_again = FR1.pixel_to_ray(tr.uv_b[ok], normalize=False) * tr.z_b[ok, None]
    P_a_again = geom.T_camB_camA.inverse().transform(P_b_again)
    assert np.allclose(P_a_again, tr.P_a[ok], atol=1e-7)


def test_true_matches_are_correct_and_shuffled_ones_are_wrong():
    geom = stereo_pair()
    uv_a = grid()
    uv_b = uv_a - np.array([25.0, 0.0])
    ident = np.stack([np.arange(len(uv_a))] * 2, axis=1)

    lab = label_matches(uv_a, uv_b, ident, geom, thresh_px=1.0)
    assert (lab.labels == CORRECT).all()
    assert np.nanmax(lab.error_px) < 1e-9

    rolled = ident.copy()
    rolled[:, 1] = np.roll(rolled[:, 1], 1)       # every match points to a neighbour
    lab = label_matches(uv_a, uv_b, rolled, geom, thresh_px=1.0)
    assert (lab.labels == WRONG).all()
    assert lab.precision == 0.0


def test_inverted_transform_is_caught_by_the_labels():
    """Passing T_camA_camB instead of T_camB_camA predicts the shift with the
    wrong sign: 50 px off, every true match labelled WRONG. This is what the
    bug looks like in practice -- a matcher that appears useless."""
    good = stereo_pair()
    bad = PairGeometry(IDEAL, good.T_camB_camA.inverse(), good.depth_a, good.depth_b)
    uv_a = grid()
    uv_b = uv_a - np.array([25.0, 0.0])
    ident = np.stack([np.arange(len(uv_a))] * 2, axis=1)
    lab = label_matches(uv_a, uv_b, ident, bad, thresh_px=3.0)
    assert (lab.labels == WRONG).all()
    assert np.allclose(lab.error_px, 50.0)


def test_threshold_is_inclusive_and_in_pixels():
    geom = stereo_pair()
    uv_a = np.array([[300.0, 200.0]])
    for off, want in [(2.99, CORRECT), (3.0, CORRECT), (3.01, WRONG)]:
        uv_b = np.array([[275.0 + off, 200.0]])
        assert label_matches(uv_a, uv_b, [[0, 0]], geom, thresh_px=3.0).labels[0] == want


def test_missing_depth_gives_unknown_not_wrong():
    geom = stereo_pair()
    geom.depth_a[190:211, 290:311] = np.nan
    uv_a = np.array([[300.0, 200.0], [400.0, 300.0]])
    uv_b = uv_a - np.array([25.0, 0.0])
    lab = label_matches(uv_a, uv_b, [[0, 0], [1, 1]], geom)
    assert lab.labels.tolist() == [UNKNOWN, CORRECT]
    assert lab.precision == 1.0                     # UNKNOWN excluded, not counted wrong
    assert transfer_keypoints(uv_a, geom).reason[0] == NO_DEPTH_A


def test_hole_next_to_the_keypoint_also_gives_unknown():
    geom = stereo_pair()
    geom.depth_a[200, 301] = np.nan                 # one neighbour of (300, 200) missing
    assert transfer_keypoints(np.array([[300.0, 200.0]]), geom).reason[0] == NO_DEPTH_A


def test_depth_edge_gives_unknown():
    """Keypoint on the boundary between a box at 1 m and the wall at 2 m. Its
    depth pixel could belong to either surface, and the predicted shift differs
    by a factor of two (50 px vs 25 px). Neither verdict would be trustworthy."""
    geom = stereo_pair()
    geom.depth_a[:, :300] = 1.0
    uv_a = np.array([[300.0, 200.0], [200.0, 200.0], [400.0, 200.0]])
    tr = transfer_keypoints(uv_a, geom)
    assert tr.reason.tolist() == [DEPTH_EDGE_A, OK, OK]
    assert tr.uv_b[1, 0] == pytest.approx(200.0 - 50.0)   # box: 1 m -> 50 px
    assert tr.uv_b[2, 0] == pytest.approx(400.0 - 25.0)   # wall: 2 m -> 25 px


def test_out_of_view_and_behind():
    geom = stereo_pair()
    tr = transfer_keypoints(np.array([[10.0, 200.0]]), geom)   # shifts to u = -15
    assert tr.reason[0] == OUT_OF_VIEW_B

    # B turned around 180 deg about y: everything A sees is behind B.
    flip = SE3(exp_so3(np.array([0.0, np.pi, 0.0])), np.zeros(3), "camB", "camA")
    g2 = PairGeometry(IDEAL, flip, wall(IDEAL), None)
    uv_a = np.array([[300.0, 200.0]])
    assert transfer_keypoints(uv_a, g2).reason[0] == BEHIND_B
    lab = label_matches(uv_a, uv_a, [[0, 0]], g2)
    assert lab.labels[0] == WRONG and np.isinf(lab.error_px[0])


def test_occlusion_removes_partner_but_not_the_label():
    """A pole in front of the wall, seen only in B's depth. The A-point behind
    it is hidden in B, so it must not count towards recall. A match to the
    pixel where it projects is still pose-consistent, so it stays CORRECT."""
    geom = stereo_pair()
    geom.depth_b[:, 270:281] = 0.8                 # occluder covering u in [270, 280]
    uv_a = np.array([[300.0, 200.0], [400.0, 200.0]])    # -> 275 (hidden), 375 (visible)
    uv_b = uv_a - np.array([25.0, 0.0])
    tr = transfer_keypoints(uv_a, geom)
    assert tr.reason.tolist() == [OCCLUDED_B, OK]

    par = ground_truth_partners(uv_a, uv_b, geom)
    assert par.partner.tolist() == [-1, 1]
    lab = label_matches(uv_a, uv_b, [[0, 0], [1, 1]], geom)
    assert lab.labels.tolist() == [CORRECT, CORRECT]


def test_partner_requires_a_detection_in_b():
    """A visible point with no keypoint near its prediction is a detector miss,
    not a matcher miss: it must not enter the recall denominator."""
    geom = stereo_pair()
    uv_a = np.array([[300.0, 200.0], [400.0, 300.0]])
    uv_b = np.array([[275.0, 200.0], [100.0, 100.0]])      # nothing near (375, 300)
    par = ground_truth_partners(uv_a, uv_b, geom, thresh_px=3.0)
    assert par.partner.tolist() == [0, -1]
    assert par.n_matchable == 1


def test_precision_and_recall_arithmetic():
    """10 matchable keypoints; matcher returns 6 correct, 2 wrong, 1 unknown."""
    geom = stereo_pair()
    uv_a = grid(n=4)[:11]                         # 11 keypoints
    geom.depth_a[np.rint(uv_a[10, 1]).astype(int), np.rint(uv_a[10, 0]).astype(int)] = np.nan
    uv_b = uv_a - np.array([25.0, 0.0])
    matches = [[k, k] for k in range(6)] + [[6, 7], [7, 6]] + [[10, 10]]
    r = evaluate_matches(uv_a, uv_b, matches, geom, thresh_px=3.0)
    assert (r["n_correct"], r["n_wrong"], r["n_unknown"]) == (6, 2, 1)
    assert r["n_matchable"] == 10
    assert r["precision"] == pytest.approx(6 / 8)
    assert r["recall"] == pytest.approx(6 / 10)


def test_duplicate_assignment_of_an_a_keypoint_is_rejected():
    geom = stereo_pair()
    uv = grid(n=3)
    with pytest.raises(ValueError, match="more than once"):
        label_matches(uv, uv, [[0, 0], [0, 1]], geom)


def test_frame_labels_reject_a_self_transform():
    with pytest.raises(ValueError):
        PairGeometry(IDEAL, SE3(np.eye(3), np.zeros(3), "camA", "camA"), wall(IDEAL))


# ---------------------------------------------------------------- interpolation
def _two_sample_traj(theta=0.2):
    P0 = SE3(np.eye(3), [0.0, 0.0, 0.0], "world", "cam")
    P1 = SE3(exp_so3(np.array([0.0, 0.0, theta])), [0.02, -0.04, 0.0], "world", "cam")
    return Trajectory(np.array([10.0, 10.01]), [P0, P1])


def test_interpolation_hits_samples_and_halves_the_motion():
    traj = _two_sample_traj()
    assert np.allclose(interpolate_pose(traj, 10.0).matrix(), traj.poses[0].matrix())
    assert np.allclose(interpolate_pose(traj, 10.01).matrix(), traj.poses[1].matrix())
    mid = interpolate_pose(traj, 10.005)
    assert np.allclose(mid.R, exp_so3(np.array([0.0, 0.0, 0.1])), atol=1e-12)
    assert np.allclose(mid.t, [0.01, -0.02, 0.0], atol=1e-12)


def test_interpolation_refuses_to_extrapolate_or_bridge_gaps():
    traj = _two_sample_traj()
    assert interpolate_pose(traj, 9.99) is None
    assert interpolate_pose(traj, 10.02) is None
    assert interpolate_pose(traj, 10.005, max_gap=0.005) is None


def test_camera_speed_of_a_constant_rotation():
    from anchor.data.trajectory import camera_speed
    ts = np.arange(0, 1.0, 0.01)
    w = np.radians(30.0)                                  # 30 deg/s about z
    poses = [SE3(exp_so3(np.array([0, 0, w * t])), [0.5 * t, 0, 0], "world", "cam") for t in ts]
    lin, ang = camera_speed(Trajectory(ts, poses), 0.5)
    assert lin == pytest.approx(0.5, rel=1e-9)
    assert ang == pytest.approx(w, rel=1e-9)
