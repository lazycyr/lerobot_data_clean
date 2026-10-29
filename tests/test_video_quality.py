from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from dclean.dataset import EpisodeMeta
from dclean.video_quality import SampleSeries, analyze_episode, check_video_quality, extract_samples


def _episode(length: int = 300) -> EpisodeMeta:
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


def _series(n: int = 50) -> SampleSeries:
    return SampleSeries(
        times=np.arange(n, dtype=np.float64) / 5.0,
        mean_luma=np.full(n, 150.0),
        std_luma=np.full(n, 40.0),
        blur=np.full(n, 1000.0),
        frame_diff=np.full(n, 2.0),
        dark_fraction=np.zeros(n),
        bright_fraction=np.zeros(n),
        fps=5.0,
    )


def test_analyze_episode_flags_black_blur_freeze_and_exposure(cfg):
    series = _series()
    series.mean_luma[10:15] = 0.0
    series.std_luma[10:15] = 0.0
    series.blur[20:25] = 100.0
    series.frame_diff[30:35] = 0.0
    series.bright_fraction[40:45] = 1.0

    issues, row = analyze_episode(
        _episode(),
        "cam",
        series,
        np.arange(50),
        from_s=0.0,
        to_s=10.0,
        blur_threshold=500.0,
        cfg=cfg,
    )
    ids = {issue.issue_id for issue in issues}
    assert {
        "l3.video_black",
        "l3.video_blur",
        "l3.video_frozen",
        "l3.video_overexposed",
    } <= ids
    assert row.status == "fail"
    assert row.black_ratio > 0 and row.blur_low_ratio > 0 and row.freeze_ratio > 0
    black = next(issue for issue in issues if issue.issue_id == "l3.video_black")
    assert 50 <= black.frame_start <= 70
    assert black.frame_end >= black.frame_start


def test_extract_samples_black_and_motion(tmp_path, cfg):
    black = tmp_path / "black.mp4"
    moving = tmp_path / "moving.mp4"
    for path, source in [
        (black, "color=c=black:s=64x64:r=30"),
        (moving, "testsrc2=s=64x64:r=30"),
    ]:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                source,
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )

    black_series = extract_samples(black, cfg, 30.0)
    assert black_series.times.size == 30
    assert black_series.mean_luma.max() < 2
    assert black_series.std_luma.max() < 1

    moving_series = extract_samples(moving, cfg, 30.0)
    assert moving_series.times.size == 30
    assert moving_series.blur.max() > 0
    assert moving_series.frame_diff[1:].max() > 0


def test_check_video_quality_synthetic(loaded, synthetic_root, cfg):
    info, tasks, episodes, arrays = loaded
    dataset_issues, episode_issues, rows, calibration = check_video_quality(
        synthetic_root, info, episodes, cfg
    )
    # ep9's wrist_right metadata points to a missing file, so that camera has
    # no quality row; the missing file itself is an L0 issue.
    assert len(rows) == len(episodes) * 4 - 1
    assert calibration["enabled"] is True
    assert len(calibration["cameras"]) == 4
    assert any(issue.issue_id == "l3.video_quality_baseline" for issue in dataset_issues)
