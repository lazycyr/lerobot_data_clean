from __future__ import annotations

import pytest

from dclean.video import check_video, parse_rate, probe_video


def test_parse_rate():
    assert parse_rate("30/1") == pytest.approx(30.0)
    assert parse_rate("30000/1001") == pytest.approx(29.97002997)
    assert parse_rate("0/0") is None
    assert parse_rate("N/A") is None
    assert parse_rate(None) is None


def test_probe_h264(synthetic_root):
    path = synthetic_root / "videos" / "observation.images.head_left" / "chunk-000" / "file-000.mp4"
    probe = probe_video(path)
    assert probe.error is None
    assert probe.codec == "h264"
    assert probe.width == 64 and probe.height == 64
    assert probe.fps == pytest.approx(30.0, abs=1e-3)
    assert probe.nb_frames == 300
    assert probe.duration == pytest.approx(10.0, abs=1e-2)


def test_probe_av1(synthetic_root):
    path = (
        synthetic_root
        / "videos"
        / "observation.images.wrist_right"
        / "chunk-000"
        / "file-001.mp4"
    )
    probe = probe_video(path)
    assert probe.error is None
    assert probe.codec == "av1"
    assert probe.nb_frames == 120


def test_check_video_metadata(loaded, synthetic_root, cfg):
    info, tasks, episodes, arrays = loaded
    dataset_issues, episode_issues = check_video(
        synthetic_root,
        info,
        episodes,
        all_episodes=episodes,
        arrays_by_episode=arrays,
        cfg=cfg,
    )
    dataset_ids = {issue.issue_id for issue in dataset_issues}
    assert "l3.video_codec_mismatch" in dataset_ids
    assert "l1.video_frame_count" not in dataset_ids

    ep8_ids = {issue.issue_id for issue in episode_issues if issue.episode_index == 8}
    assert "l1.video_segment_duration" in ep8_ids
    assert not any(
        issue.issue_id == "l1.video_bounds" and issue.episode_index == 0
        for issue in episode_issues
    )
    # wrist_right episode 5 uses file-001 at offset 0 while head_left uses
    # file-000 at 5s; both are valid and should not be flagged.
    assert not any(
        issue.episode_index == 5 and issue.issue_id == "l1.video_bounds"
        for issue in episode_issues
    )


def test_check_video_subset_uses_all_episodes_for_file_counts(loaded, synthetic_root, cfg):
    info, tasks, episodes, arrays = loaded
    selected = episodes[:3]
    dataset_issues, episode_issues = check_video(
        synthetic_root,
        info,
        selected,
        all_episodes=episodes,
        arrays_by_episode={ep.episode_index: arrays[ep.episode_index] for ep in selected},
        cfg=cfg,
    )
    # The head_left file holds all 10 episodes; a 3-episode subset must not
    # produce a bogus frame-count mismatch.
    assert not any(issue.issue_id == "l1.video_frame_count" for issue in dataset_issues)
    assert not any(issue.issue_id == "l3.video_codec_mismatch" for issue in dataset_issues)
