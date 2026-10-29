from __future__ import annotations

import numpy as np
import pytest

from dclean.av_sync import compare_segment, extract_packet_pts
from dclean.dataset import EpisodeMeta


def _episode(length: int = 30) -> EpisodeMeta:
    return EpisodeMeta(
        episode_index=0,
        length=length,
        tasks=("task",),
        data_chunk_index=0,
        data_file_index=0,
        dataset_from_index=0,
        dataset_to_index=length,
        videos={},
    )


def test_compare_segment_aligned(cfg):
    pts = np.arange(30, dtype=np.float64) / 30.0
    action_ts = np.arange(30, dtype=np.float64) / 30.0
    issue, row = compare_segment(
        _episode(), "cam", pts, action_ts, 0.0, 1.0, 30.0, cfg
    )
    assert issue is None
    assert row.status == "ok"
    assert row.video_frames == 30
    assert row.start_offset_s == 0.0
    assert row.max_drift_s is not None and row.max_drift_s < 1e-9


def test_compare_segment_drift(cfg):
    pts = np.arange(30, dtype=np.float64) / 30.0 + 0.05
    action_ts = np.arange(30, dtype=np.float64) / 30.0
    issue, row = compare_segment(
        _episode(), "cam", pts, action_ts, 0.0, 1.0, 30.0, cfg
    )
    assert issue is not None
    assert issue.issue_id == "l1.av_timestamp_mismatch"
    assert issue.severity == "fail"
    assert row.status == "fail"
    assert row.start_offset_s > 0.04


def test_compare_segment_frame_count(cfg):
    pts = np.arange(29, dtype=np.float64) / 30.0
    action_ts = np.arange(30, dtype=np.float64) / 30.0
    issue, row = compare_segment(
        _episode(), "cam", pts, action_ts, 0.0, 1.0, 30.0, cfg
    )
    assert issue is not None
    assert "video_frames=29" in issue.detail
    assert row.video_frames == 29


def test_extract_packet_pts_synthetic(synthetic_root):
    path = (
        synthetic_root
        / "videos"
        / "observation.images.head_left"
        / "chunk-000"
        / "file-000.mp4"
    )
    pts, error = extract_packet_pts(path)
    assert error is None
    assert pts.size == 300
    assert pts[0] == 0.0
    assert np.median(np.diff(pts)) == pytest.approx(1.0 / 30.0, abs=1e-5)
