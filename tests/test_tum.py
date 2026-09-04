"""TUM RGB-D loader, tested on a synthetic sequence with the real file layout."""

import cv2
import numpy as np
import pytest

from anchor.data.tum import (
    TUMDataset, detect_category, read_file_list, tum_camera,
    verify_groundtruth_convention,
)
from anchor.geometry.rotations import rot_to_quat_xyzw
from anchor.geometry.synthetic import look_at

DEPTH_SCALE = 5000.0


def render_scene_depth(camera, T_world_cam, planes):
    """Ray-cast a set of world planes into a depth image, in metres.

    Needed because a *geometrically consistent* multi-view fixture is the only
    kind that can test the ground-truth frame convention: constant-depth images
    do not correspond to any single 3D scene, so reprojecting one into another
    proves nothing.

    Each plane is ``(n, d)`` with ``n . X = d``. For pixel ray ``r`` (scaled so
    ``r_z = 1``) from camera centre C, the world point is ``C + s R r``, so
    ``s = (d - n.C) / (n . R r)`` -- and because ``r_z = 1``, that ``s`` *is*
    the depth along the optical axis, which is what a depth image stores.
    """
    us, vs = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    uv = np.stack([us.ravel(), vs.ravel()], axis=1).astype(float)
    rays_world = T_world_cam.rotate(camera.pixel_to_ray(uv, normalize=False))
    C = T_world_cam.t

    best = np.full(len(uv), np.inf)
    for n, d in planes:
        n = np.asarray(n, float)
        denom = rays_world @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (d - C @ n) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        s = np.where(np.isfinite(s) & (s > 0.2) & (s < 12.0), s, np.inf)
        best = np.minimum(best, s)

    depth = best.reshape(camera.height, camera.width)
    depth[~np.isfinite(depth)] = 0.0        # no intersection -> no measurement
    return depth


def build_sequence(root, n_rgb=6, scene="planes", flat_depth_m=2.0,
                   missing_depth_frame=None, name="rgbd_dataset_freiburg1_synthetic"):
    """Write a minimal but format-accurate TUM sequence to disk.

    Colour at 30 Hz, depth at 30 Hz but offset by 8 ms (real sensors are not
    synchronised), ground truth at 100 Hz.

    ``scene="planes"`` ray-casts a floor and a wall from each camera pose, so
    the frames are views of one rigid scene. ``scene="flat"`` writes a
    fronto-parallel plane at a constant depth, which makes unprojection
    checkable in closed form but is not multi-view consistent.
    """
    seq = root / name
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)
    camera, _ = tum_camera("fr1")

    t0 = 1305031102.175304
    rgb_ts = t0 + np.arange(n_rgb) / 30.0
    depth_ts = rgb_ts + 0.008

    # Ground truth at 100 Hz, spanning the colour frames, with real motion.
    gt_ts = t0 + np.arange(0, int(n_rgb / 30.0 * 100) + 20) / 100.0

    def pose_at(k):
        """Camera pose as a function of the ground-truth sample index."""
        return look_at([1.4 - 0.008 * k, -1.1 + 0.010 * k, 1.15 + 0.002 * k],
                       [0.0, 1.4, 0.55])

    rgb_lines, depth_lines = ["# color images"], ["# depth maps"]
    for i, (tr, td) in enumerate(zip(rgb_ts, depth_ts)):
        img = np.full((480, 640, 3), 40 * (i + 1) % 255, np.uint8)
        cv2.imwrite(str(seq / "rgb" / f"{tr:.6f}.png"), img)
        rgb_lines.append(f"{tr:.6f} rgb/{tr:.6f}.png")

        if i == missing_depth_frame:
            continue
        if scene == "flat":
            depth = np.full((480, 640), float(flat_depth_m))
        else:
            # The depth exposure happens at td, so use the pose at that instant.
            depth = render_scene_depth(
                camera, pose_at(int(round((td - t0) * 100))),
                planes=[(np.array([0.0, 0.0, 1.0]), 0.0),      # floor  z = 0
                        (np.array([0.0, 1.0, 0.0]), 2.0)])     # wall   y = 2
        raw = np.clip(np.round(depth * DEPTH_SCALE), 0, 65535).astype(np.uint16)
        raw[:10, :10] = 0                       # a patch of "no measurement"
        cv2.imwrite(str(seq / "depth" / f"{td:.6f}.png"), raw)
        depth_lines.append(f"{td:.6f} depth/{td:.6f}.png")

    gt_lines = ["# ground truth trajectory"]
    for k, t in enumerate(gt_ts):
        T = pose_at(k)
        qx, qy, qz, qw = rot_to_quat_xyzw(T.R)
        gt_lines.append(f"{t:.6f} {T.t[0]:.6f} {T.t[1]:.6f} {T.t[2]:.6f} "
                        f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    (seq / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    (seq / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n")
    return seq


# --------------------------------------------------------------------- basics

def test_intrinsics_registry_matches_orbslam3_configs():
    """Regression guard on transcribed constants. These exact values are what
    ORB-SLAM3's TUM1/TUM2/TUM3 configs use; drifting from them would make the
    benchmark comparison dishonest."""
    cam, scale = tum_camera("fr1")
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (517.306408, 516.469215, 318.643040, 255.313989)
    assert np.allclose(cam.dist, [0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
    assert scale == 5000.0

    cam2, scale2 = tum_camera("fr2")
    assert (cam2.fx, cam2.cx) == (520.908620, 325.141442)
    assert scale2 == 5208.0          # NOT 5000 -- see the comment in tum.py

    cam3, scale3 = tum_camera("fr3")
    assert not cam3.has_distortion   # fr3 configs are undistorted
    assert scale3 == 5000.0


def test_unknown_camera_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        tum_camera("freiburg9")


def test_category_detection():
    assert detect_category("/data/rgbd_dataset_freiburg1_xyz") == "fr1"
    assert detect_category("/data/rgbd_dataset_freiburg3_long_office_household") == "fr3"
    with pytest.raises(ValueError, match="cannot tell which camera"):
        detect_category("/data/my_own_recording")


def test_read_file_list_rejects_malformed_lines(tmp_path):
    p = tmp_path / "rgb.txt"
    p.write_text("# header\n1.0 rgb/a.png extra\n")
    with pytest.raises(ValueError, match="expected 'timestamp filename'"):
        read_file_list(p)


# ---------------------------------------------------------------- association

def test_loads_and_associates_all_three_streams(tmp_path):
    seq = build_sequence(tmp_path)
    ds = TUMDataset(seq)
    assert ds.category == "fr1"
    assert len(ds) == 6
    for f in ds.frames:
        assert f.depth_path is not None and f.depth_path.exists()
        assert f.T_world_cam is not None
        # Real colour/depth exposure offset. Tolerance is 2e-6 s, not tighter:
        # TUM writes timestamps with 6 decimal places, so each is rounded to the
        # nearest microsecond and a difference of two of them carries up to 1 us
        # of quantisation from each.
        assert f.dt_depth == pytest.approx(0.008, abs=2e-6)
        assert abs(f.pose_timestamp - f.timestamp) <= 0.02


def test_frames_without_depth_are_dropped_or_kept_on_request(tmp_path):
    seq = build_sequence(tmp_path, missing_depth_frame=2)
    assert len(TUMDataset(seq, require_depth=True)) == 5
    lenient = TUMDataset(seq, require_depth=False)
    assert len(lenient) == 6
    assert sum(f.depth_path is None for f in lenient.frames) == 1
    assert lenient.n_rgb_total == 6


def test_trajectory_matches_the_associated_frames(tmp_path):
    ds = TUMDataset(build_sequence(tmp_path))
    traj = ds.trajectory()
    assert len(traj) == len(ds)
    assert np.allclose(traj.timestamps, [f.timestamp for f in ds.frames])


# ----------------------------------------------------------------- depth / 3D

def test_depth_is_metric_and_missing_data_is_nan(tmp_path):
    ds = TUMDataset(build_sequence(tmp_path, scene="flat", flat_depth_m=2.0))
    depth = ds.load_depth(0)
    assert depth.shape == (480, 640)
    assert np.isnan(depth[:10, :10]).all()          # the zero patch
    assert np.nanmax(depth) == pytest.approx(2.0, abs=1e-4)
    assert np.isfinite(depth[240, 320])


def test_unprojection_of_a_fronto_parallel_plane_has_constant_z(tmp_path):
    """The check that catches the ray-normalisation bug.

    Depth in an RGB-D image is depth *along the optical axis*, not distance
    along the viewing ray. If you multiply a unit-length ray by the depth
    value, every point comes out too far by 1/cos(angle off axis) -- a plane
    turns into a bowl, curving away by ~11% at the corners of a 60-degree lens.
    A correct unprojection gives every point on a fronto-parallel plane exactly
    the same z.
    """
    ds = TUMDataset(build_sequence(tmp_path, scene="flat", flat_depth_m=2.0))
    P, uv = ds.unproject(0, stride=4)
    assert len(P) > 10000
    assert np.allclose(P[:, 2], 2.0, atol=1e-4)     # flat, not bowl-shaped

    # And it really is a plane of finite extent, not a degenerate point.
    assert np.ptp(P[:, 0]) > 1.5 and np.ptp(P[:, 1]) > 1.0

    # Round trip: those 3D points must project back to the pixels they came from.
    uv_back = ds.camera.project(P, return_mask=False)
    assert np.abs(uv_back - uv).max() < 1e-6


def test_unprojection_skips_missing_depth(tmp_path):
    ds = TUMDataset(build_sequence(tmp_path, scene="flat"))
    P, uv = ds.unproject(0, stride=1)
    assert np.isfinite(P).all()
    assert len(P) == 480 * 640 - 100                # the 10x10 zero patch is gone


# ------------------------------------------------- ground-truth frame convention

def test_groundtruth_convention_is_verified_against_the_data(tmp_path):
    """The documentation and secondary sources disagree about whether TUM's
    ground truth is T_world_cam or its inverse. Rather than trusting either,
    reproject one frame's depth into another and see which convention predicts
    the depths that were actually measured.

    The residual under the correct convention is ~1 cm rather than zero, and
    that is the fixture being honest rather than an error: depth is exposed
    8 ms after colour, the loader associates ground truth to the *colour*
    timestamp, and this fixture's camera moves at 1.3 m/s. 0.008 s x 1.3 m/s
    = 1.0 cm, which is what comes out. The wrong convention is an order of
    magnitude worse, so the test still discriminates cleanly.
    """
    ds = TUMDataset(build_sequence(tmp_path, n_rgb=14))
    report = verify_groundtruth_convention(ds, i=0, j=11, stride=4)
    assert report["verdict"] == "as_T_world_cam"
    assert report["as_T_world_cam"]["median_depth_residual_m"] < 0.02
    assert report["ratio"] > 5.0          # decisive, not a coin flip


def test_missing_sequence_directory_is_an_error():
    with pytest.raises(FileNotFoundError):
        TUMDataset("/nonexistent/rgbd_dataset_freiburg1_ghost")
