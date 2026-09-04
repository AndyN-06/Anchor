"""Trajectory container, TUM-format I/O, and timestamp association."""

import numpy as np
import pytest

from anchor.data.trajectory import Trajectory, associate
from anchor.geometry.rotations import exp_so3, rot_to_quat_xyzw
from anchor.geometry.se3 import SE3

RNG = np.random.default_rng(11)


def make_trajectory(n=50, t0=0.0, dt=0.1):
    ts = t0 + np.arange(n) * dt
    poses = [SE3(exp_so3(np.array([0.01 * i, -0.02 * i, 0.005 * i])),
                 np.array([0.1 * i, np.sin(0.1 * i), 0.02 * i]), "world", "cam")
             for i in range(n)]
    return Trajectory(ts, poses, "test")


def test_tum_file_roundtrip(tmp_path):
    traj = make_trajectory()
    path = tmp_path / "traj.txt"
    traj.to_tum_file(path)
    back = Trajectory.from_tum_file(path)
    assert len(back) == len(traj)
    assert np.allclose(back.timestamps, traj.timestamps, atol=1e-6)
    assert np.allclose(back.positions, traj.positions, atol=1e-9)
    for a, b in zip(back.poses, traj.poses):
        assert np.allclose(a.R, b.R, atol=1e-9)


def test_reader_handles_comments_blank_lines_and_whitespace(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(
        "# comment header\n"
        "\n"
        "1305031102.1758 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"
        "   \n"
        "1305031102.2118\t1.1\t2.0 3.0 0.0 0.0 0.0 1.0   # trailing comment\n"
    )
    traj = Trajectory.from_tum_file(path)
    assert len(traj) == 2
    assert np.allclose(traj.poses[0].R, np.eye(3))
    assert np.allclose(traj.positions[1], [1.1, 2.0, 3.0])


def test_reader_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1.0 1.0 2.0 3.0 0.0 0.0 0.0\n")     # 7 fields, not 8
    with pytest.raises(ValueError, match="expected 8 fields"):
        Trajectory.from_tum_file(path)


def test_reader_rejects_non_unit_quaternion(tmp_path):
    """A file whose columns are in the wrong order usually shows up here first:
    the four 'quaternion' columns will not have unit norm."""
    path = tmp_path / "bad.txt"
    path.write_text("1.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0\n")
    with pytest.raises(ValueError, match="quaternion norm"):
        Trajectory.from_tum_file(path)


def test_reader_sorts_unordered_files(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("2.0 0 0 0 0 0 0 1\n1.0 1 0 0 0 0 0 1\n")
    traj = Trajectory.from_tum_file(path)
    assert traj.timestamps.tolist() == [1.0, 2.0]
    assert np.allclose(traj.positions[0], [1, 0, 0])


def test_quaternion_written_in_xyzw_order(tmp_path):
    """Guard against silently writing wxyz: the file must be readable by TUM's
    own evaluation scripts, which assume scalar-last."""
    R = exp_so3(np.array([0.4, -0.9, 0.2]))
    traj = Trajectory(np.array([1.0]), [SE3(R, np.zeros(3))])
    path = tmp_path / "t.txt"
    traj.to_tum_file(path)
    fields = [f for f in path.read_text().splitlines() if not f.startswith("#")][0].split()
    assert np.allclose([float(x) for x in fields[4:8]], rot_to_quat_xyzw(R), atol=1e-8)


def test_constructor_rejects_mismatched_lengths_and_unsorted_input():
    with pytest.raises(ValueError, match="timestamps but"):
        Trajectory(np.array([0.0, 1.0]), [SE3.identity()])
    with pytest.raises(ValueError, match="non-decreasing"):
        Trajectory(np.array([1.0, 0.0]), [SE3.identity(), SE3.identity()])


def test_path_length_and_duration():
    poses = [SE3(np.eye(3), np.array([float(i), 0.0, 0.0])) for i in range(5)]
    traj = Trajectory(np.arange(5) * 0.5, poses)
    assert traj.path_length() == pytest.approx(4.0)
    assert traj.duration == pytest.approx(2.0)


def test_transformed_moves_positions_and_rotations_consistently():
    traj = make_trajectory(10)
    T = SE3(exp_so3(np.array([0.3, 0.2, -0.5])), np.array([1.0, -2.0, 0.5]))
    moved = traj.transformed(T, scale=2.0)
    assert np.allclose(moved.positions, 2.0 * (traj.positions @ T.R.T) + T.t)
    assert np.allclose(moved.poses[3].R, T.R @ traj.poses[3].R)
    # Scale must NOT touch rotation -- a rotation has no size.
    assert np.isclose(np.linalg.det(moved.poses[3].R), 1.0)


# ------------------------------------------------------------------ association

def test_association_is_one_to_one():
    """Two colour frames must never both claim the same ground-truth pose --
    that would duplicate it in the error statistics and quietly bias the RMSE."""
    a = np.array([0.00, 0.01, 0.50])
    b = np.array([0.005, 0.60])
    pairs = associate(a, b, max_diff=0.02)
    assert len(pairs) == 1                       # only the closest of a[0], a[1] wins
    assert pairs[0].tolist() == [0, 0]
    assert len(set(pairs[:, 0])) == len(pairs)
    assert len(set(pairs[:, 1])) == len(pairs)


def test_association_picks_the_closest_pair_not_the_first():
    a = np.array([0.010])
    b = np.array([0.000, 0.009, 0.025])
    pairs = associate(a, b, max_diff=0.02)
    assert pairs.tolist() == [[0, 1]]            # 0.009 is closest to 0.010


def test_association_respects_max_diff():
    a, b = np.array([0.0]), np.array([0.05])
    assert len(associate(a, b, max_diff=0.02)) == 0
    assert len(associate(a, b, max_diff=0.10)) == 1


def test_association_handles_a_constant_clock_offset():
    a = np.arange(10) * 0.1
    b = a + 7.0
    assert len(associate(a, b, max_diff=0.02)) == 0
    assert len(associate(a, b, max_diff=0.02, offset=-7.0)) == 10


def test_association_on_realistic_rates():
    """30 Hz colour against 100 Hz mocap: every frame should find a partner."""
    rgb = np.arange(300) / 30.0
    gt = np.arange(1000) / 100.0 + 0.0031
    pairs = associate(rgb, gt, max_diff=0.02)
    assert len(pairs) == 300
    dt = np.abs(rgb[pairs[:, 0]] - gt[pairs[:, 1]])
    assert dt.max() <= 0.02


def test_association_of_empty_inputs():
    assert associate(np.array([]), np.array([1.0])).shape == (0, 2)
