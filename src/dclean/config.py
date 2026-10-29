"""Configuration loading for dclean.

TOML is used so the project can rely on the standard library ``tomllib`` and
does not need PyYAML. Unknown sections/keys are rejected to catch typos.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    fps: float = 30.0

    # L1
    dt_abs_tol_s: float = 1e-4
    max_gap_frames: int = 1

    # L2
    joint_dims: tuple[int, int] = (0, 17)
    discrete_dims: tuple[int, ...] = (18, 19)
    constant_std_eps: float = 1e-6
    constant_global_fraction: float = 0.95
    step_scale_floor: float = 0.01
    step_abs_floor: float = 0.1
    step_z: float = 6.0
    step_fail_count: int = 50
    action_state_p99_warn: float = 0.02
    action_state_max_warn: float = 0.1
    action_state_frame_tol: float = 0.05

    # L2 trajectory deviation (median trajectory baseline)
    trajectory_dims: tuple[int, int] = (0, 17)
    trajectory_min_episodes: int = 10
    trajectory_smooth_frames: int = 15
    trajectory_min_segment_frames: int = 15
    trajectory_merge_gap_frames: int = 5
    trajectory_scale_floor: float = 0.01
    trajectory_z_warn: float = 10.0
    trajectory_z_fail: float = 20.0
    trajectory_abs_floor: float = 0.2
    trajectory_abs_fail: float = 1.0

    # L3
    codec_mismatch_status: str = "info"
    frame_count_tol: int = 1

    # Action/video PTS alignment (read-only check)
    av_sync_enabled: bool = True
    av_sync_tol_frames: float = 1.0
    av_sync_warn_frames: float = 0.5
    av_sync_pts_gap_tol_frames: float = 0.5

    # L4 video frame quality (sampled grayscale frames)
    video_quality_enabled: bool = True
    video_quality_full_frame: bool = True
    video_quality_sample_fps: float = 5.0
    video_quality_width: int = 160
    video_quality_height: int = 120
    video_black_luma: float = 20.0
    video_black_std: float = 5.0
    video_dark_luma: float = 40.0
    video_bright_luma: float = 245.0
    video_bright_fraction: float = 0.5
    video_freeze_diff: float = 0.5
    video_freeze_min_duration_s: float = 0.8
    video_event_min_duration_s: float = 0.3
    video_blur_ratio_threshold: float = 0.5
    video_blur_min_texture_std: float = 10.0
    video_blur_min_duration_s: float = 0.5

    # Output
    summary_max_episodes: int = 50
    max_examples_in_detail: int = 3


_SECTION_KEYS: dict[str, dict[str, str]] = {
    "fps": {"expected": "fps"},
    "l1": {
        "dt_abs_tol_s": "dt_abs_tol_s",
        "max_gap_frames": "max_gap_frames",
    },
    "l2": {
        "joint_dims": "joint_dims",
        "discrete_dims": "discrete_dims",
        "constant_std_eps": "constant_std_eps",
        "constant_global_fraction": "constant_global_fraction",
        "step_scale_floor": "step_scale_floor",
        "step_abs_floor": "step_abs_floor",
        "step_z": "step_z",
        "step_fail_count": "step_fail_count",
        "action_state_p99_warn": "action_state_p99_warn",
        "action_state_max_warn": "action_state_max_warn",
        "action_state_frame_tol": "action_state_frame_tol",
        "trajectory_dims": "trajectory_dims",
        "trajectory_min_episodes": "trajectory_min_episodes",
        "trajectory_smooth_frames": "trajectory_smooth_frames",
        "trajectory_min_segment_frames": "trajectory_min_segment_frames",
        "trajectory_merge_gap_frames": "trajectory_merge_gap_frames",
        "trajectory_scale_floor": "trajectory_scale_floor",
        "trajectory_z_warn": "trajectory_z_warn",
        "trajectory_z_fail": "trajectory_z_fail",
        "trajectory_abs_floor": "trajectory_abs_floor",
        "trajectory_abs_fail": "trajectory_abs_fail",
    },
    "l3": {
        "codec_mismatch_status": "codec_mismatch_status",
        "frame_count_tol": "frame_count_tol",
    },
    "av_sync": {
        "enabled": "av_sync_enabled",
        "tol_frames": "av_sync_tol_frames",
        "warn_frames": "av_sync_warn_frames",
        "pts_gap_tol_frames": "av_sync_pts_gap_tol_frames",
    },
    "video_quality": {
        "enabled": "video_quality_enabled",
        "full_frame": "video_quality_full_frame",
        "sample_fps": "video_quality_sample_fps",
        "width": "video_quality_width",
        "height": "video_quality_height",
        "black_luma": "video_black_luma",
        "black_std": "video_black_std",
        "dark_luma": "video_dark_luma",
        "bright_luma": "video_bright_luma",
        "bright_fraction": "video_bright_fraction",
        "freeze_diff": "video_freeze_diff",
        "freeze_min_duration_s": "video_freeze_min_duration_s",
        "event_min_duration_s": "video_event_min_duration_s",
        "blur_ratio_threshold": "video_blur_ratio_threshold",
        "blur_min_texture_std": "video_blur_min_texture_std",
        "blur_min_duration_s": "video_blur_min_duration_s",
    },
    "output": {
        "summary_max_episodes": "summary_max_episodes",
        "max_examples_in_detail": "max_examples_in_detail",
    },
}

_TUPLE_FIELDS = {"joint_dims", "discrete_dims", "trajectory_dims"}


def _validate(cfg: Config) -> None:
    if cfg.fps <= 0:
        raise ValueError("fps must be positive")
    if cfg.dt_abs_tol_s <= 0:
        raise ValueError("dt_abs_tol_s must be positive")
    if cfg.max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")
    lo, hi = cfg.joint_dims
    if lo < 0 or hi < lo:
        raise ValueError("joint_dims must be [start, end] with 0 <= start <= end")
    if any(d < 0 for d in cfg.discrete_dims):
        raise ValueError("discrete_dims must be non-negative")
    if cfg.constant_std_eps < 0 or cfg.step_scale_floor <= 0:
        raise ValueError("invalid numeric threshold")
    if cfg.step_z <= 0 or cfg.step_abs_floor <= 0:
        raise ValueError("invalid step threshold")
    if not 0 < cfg.constant_global_fraction <= 1:
        raise ValueError("constant_global_fraction must be in (0, 1]")
    tlo, thi = cfg.trajectory_dims
    if tlo < 0 or thi < tlo:
        raise ValueError("trajectory_dims must be [start, end] with 0 <= start <= end")
    if cfg.trajectory_min_episodes <= 0:
        raise ValueError("trajectory_min_episodes must be positive")
    if cfg.trajectory_smooth_frames <= 0 or cfg.trajectory_min_segment_frames <= 0:
        raise ValueError("trajectory smooth/segment windows must be positive")
    if cfg.trajectory_merge_gap_frames < 0:
        raise ValueError("trajectory_merge_gap_frames must be >= 0")
    if cfg.trajectory_scale_floor <= 0:
        raise ValueError("trajectory_scale_floor must be positive")
    if cfg.trajectory_z_warn <= 0 or cfg.trajectory_z_fail <= cfg.trajectory_z_warn:
        raise ValueError("require 0 < trajectory_z_warn < trajectory_z_fail")
    if cfg.trajectory_abs_floor <= 0 or cfg.trajectory_abs_fail <= cfg.trajectory_abs_floor:
        raise ValueError("require 0 < trajectory_abs_floor < trajectory_abs_fail")
    if cfg.av_sync_tol_frames <= 0 or cfg.av_sync_warn_frames <= 0:
        raise ValueError("av_sync frame tolerances must be positive")
    if cfg.av_sync_pts_gap_tol_frames <= 0:
        raise ValueError("av_sync_pts_gap_tol_frames must be positive")
    if cfg.video_quality_sample_fps <= 0:
        raise ValueError("video_quality_sample_fps must be positive")
    if cfg.video_quality_width <= 0 or cfg.video_quality_height <= 0:
        raise ValueError("video_quality width/height must be positive")
    if not 0 <= cfg.video_black_luma <= 255 or not 0 <= cfg.video_dark_luma <= 255:
        raise ValueError("video luma thresholds must be in [0, 255]")
    if not 0 <= cfg.video_bright_luma <= 255:
        raise ValueError("video_bright_luma must be in [0, 255]")
    if not 0 <= cfg.video_bright_fraction <= 1:
        raise ValueError("video_bright_fraction must be in [0, 1]")
    if cfg.video_blur_ratio_threshold <= 0:
        raise ValueError("video_blur_ratio_threshold must be positive")
    for name in (
        "video_freeze_min_duration_s",
        "video_event_min_duration_s",
        "video_blur_min_duration_s",
    ):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.codec_mismatch_status not in {"info", "warn", "fail"}:
        raise ValueError("codec_mismatch_status must be info/warn/fail")
    if cfg.summary_max_episodes <= 0:
        raise ValueError("summary_max_episodes must be positive")


def load_config(path: Path | None = None) -> Config:
    """Load defaults, optionally overridden by a TOML file."""
    kwargs: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for section, values in data.items():
            if section not in _SECTION_KEYS:
                raise ValueError(f"unknown config section: {section!r}")
            if not isinstance(values, dict):
                raise ValueError(f"config section {section!r} must be a table")
            for key, value in values.items():
                if key not in _SECTION_KEYS[section]:
                    raise ValueError(f"unknown key {section}.{key}")
                name = _SECTION_KEYS[section][key]
                if name in _TUPLE_FIELDS:
                    value = tuple(value)
                kwargs[name] = value

    cfg = Config(**kwargs)
    _validate(cfg)
    return cfg


def config_fields() -> set[str]:
    return {f.name for f in fields(Config)}
