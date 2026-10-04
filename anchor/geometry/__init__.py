"""Geometric primitives: rotations, rigid transforms, camera models, synthetic data."""

from .camera import PinholeCamera
from .rotations import (
    angle_between_rotations,
    exp_so3,
    is_rotation,
    log_so3,
    project_to_so3,
    quat_wxyz_to_rot,
    quat_xyzw_to_rot,
    rot_to_quat_wxyz,
    rot_to_quat_xyzw,
    skew,
    unskew,
)
from .se3 import SE3, camera_center, pose_error, relative_pose
from .synthetic import (
    Observations,
    SyntheticScene,
    look_at,
    make_scene,
    observe,
    orbit_poses,
    random_point_cloud,
)

__all__ = [
    "PinholeCamera", "SE3", "camera_center", "pose_error", "relative_pose",
    "skew", "unskew", "exp_so3", "log_so3", "is_rotation", "project_to_so3",
    "quat_wxyz_to_rot", "rot_to_quat_wxyz", "quat_xyzw_to_rot", "rot_to_quat_xyzw",
    "angle_between_rotations",
    "look_at", "random_point_cloud", "orbit_poses", "observe", "make_scene",
    "Observations", "SyntheticScene",
]
