
from __future__ import annotations


def render_pose_frame_overlay(*args, **kwargs):
    from src.visualization.pose.debug import render_pose_frame_overlay as _impl

    return _impl(*args, **kwargs)


def render_pose_frame_range_overlay(*args, **kwargs):
    from src.visualization.pose.debug import render_pose_frame_range_overlay as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "render_pose_frame_overlay",
    "render_pose_frame_range_overlay",
]

