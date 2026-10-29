from __future__ import annotations

import csv
import json
from collections import defaultdict

from dclean.cli import main
from tests.fixtures import snapshot_tree


EXPECTED = {
    2: {"l2.nan_inf"},
    3: {
        "l1.timestamp_duplicate",
        "l1.timestamp_nonmonotonic",
        "l1.av_timestamp_mismatch",
    },
    4: {"l1.timestamp_gap", "l1.av_timestamp_mismatch"},
    5: {"l2.joint_step"},
    6: {"l2.constant_channel"},
    7: {"l2.action_state_residual"},
    8: {"l1.video_segment_duration", "l1.av_timestamp_mismatch"},
    9: {"l0.global_index", "l0.missing_referenced_file"},
}


def _read_issues(out) -> dict[int, dict[str, dict]]:
    rows = json.loads((out / "episode_quality.json").read_text(encoding="utf-8"))
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["episode_index"])][row["issue_id"]] = row
    return grouped


def test_full_check_smoke(synthetic_root, synthetic_snapshot, tmp_path):
    out = tmp_path / "check"
    assert main(
        ["check", "--dataset", str(synthetic_root), "--out", str(out), "--full-report", "--no-progress"]
    ) == 0

    grouped = _read_issues(out)
    assert set(grouped) == set(EXPECTED)
    for episode, issue_ids in EXPECTED.items():
        assert issue_ids <= set(grouped[episode])
    assert 0 not in grouped and 1 not in grouped

    nan = grouped[2]["l2.nan_inf"]
    assert nan["frame_start"] == 10 and nan["frame_end"] == 20
    nonmono = grouped[3]["l1.timestamp_nonmonotonic"]
    assert nonmono["frame_start"] == 14
    duplicate = grouped[3]["l1.timestamp_duplicate"]
    assert duplicate["frame_start"] == 21
    assert grouped[4]["l1.timestamp_gap"]["frame_start"] == 7
    step = grouped[5]["l2.joint_step"]
    assert step["frame_start"] <= 11 <= step["frame_end"]
    constant = grouped[6]["l2.constant_channel"]
    assert constant["frame_start"] == 0 and constant["frame_end"] == 29
    residual = grouped[7]["l2.action_state_residual"]
    assert residual["frame_end"] == 9
    assert grouped[8]["l1.video_segment_duration"]["frame_start"] == 0
    assert grouped[9]["l0.global_index"]["frame_start"] == 0

    csv_path = out / "episode_quality.csv"
    assert csv_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
    assert reader.fieldnames == [
        "episode_index",
        "issue_id",
        "severity",
        "frame_start",
        "frame_end",
        "time_start_s",
        "time_end_s",
        "event_count",
        "detail",
        "action",
    ]
    assert len(csv_rows) == 13
    assert (out / "episode_quality.json").is_file()
    assert not (out / "episode_quality.parquet").exists()

    trajectory_csv = out / "trajectory_scores.csv"
    assert trajectory_csv.is_file()
    with trajectory_csv.open(newline="", encoding="utf-8") as handle:
        trajectory_rows = list(csv.DictReader(handle))
    assert len(trajectory_rows) == 10
    assert "max_smoothed_score" in trajectory_rows[0]
    calibration = json.loads(
        (out / "trajectory_calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["n_episodes"] == 10
    # ep2 has NaN, leaving 9 finite episodes, below trajectory_min_episodes=10.
    assert calibration["computed_episodes"] == 0
    assert calibration["suggested_threshold"] is None

    video_csv = out / "video_quality.csv"
    assert video_csv.is_file()
    with video_csv.open(newline="", encoding="utf-8") as handle:
        video_rows = list(csv.DictReader(handle))
    # ep9's wrist_right video is missing, so 39 of 40 camera/episode rows exist.
    assert len(video_rows) == 39
    video_calibration = json.loads(
        (out / "video_quality_calibration.json").read_text(encoding="utf-8")
    )
    assert len(video_calibration["cameras"]) == 4

    av_sync_csv = out / "av_sync.csv"
    assert av_sync_csv.is_file()
    with av_sync_csv.open(newline="", encoding="utf-8") as handle:
        av_rows = list(csv.DictReader(handle))
    assert len(av_rows) == 39
    assert sum(row["status"] != "ok" for row in av_rows) == 9

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Checked **10** episode(s)" in summary
    assert "**2** clean" in summary
    assert "l3.video_codec_mismatch" in summary
    assert "l0.orphan_file" in summary
    assert "## Trajectory calibration" in summary

    # Read-only guarantee: no source file may be created, touched or removed.
    assert snapshot_tree(synthetic_root) == synthetic_snapshot


def test_default_report_is_concise(synthetic_root, tmp_path):
    out = tmp_path / "check_concise"
    assert main(
        [
            "check",
            "--dataset",
            str(synthetic_root),
            "--out",
            str(out),
            "--no-video-quality",
            "--no-progress",
        ]
    ) == 0
    files = {path.name for path in out.iterdir() if path.is_file()}
    assert files == {"summary.md", "episode_quality.csv"}
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Report detail: `main`" in summary
    assert "--full-report" in summary
    assert "`episode_quality.csv` contains only problematic episodes" in summary


def test_episode_subset(synthetic_root, tmp_path):
    out = tmp_path / "check_subset"
    assert main(
        [
            "check",
            "--dataset",
            str(synthetic_root),
            "--out",
            str(out),
            "--episodes",
            "0-2",
            "--full-report",
            "--no-progress",
        ]
    ) == 0
    grouped = _read_issues(out)
    assert set(grouped) == {2}
    assert "l2.nan_inf" in grouped[2]
    calibration = json.loads(
        (out / "trajectory_calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["n_episodes"] == 3
    assert calibration["computed_episodes"] == 0
    with (out / "video_quality.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12
    with (out / "av_sync.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Checked **3** episode(s)" in summary
    # ep9's missing video is outside the selected scope and must not be reported.
    assert "l0.missing_referenced_file" not in summary


def test_no_video_skips_video_checks(synthetic_root, tmp_path):
    out = tmp_path / "check_novideo"
    assert main(
        [
            "check",
            "--dataset",
            str(synthetic_root),
            "--out",
            str(out),
            "--no-video",
            "--full-report",
            "--no-progress",
        ]
    ) == 0
    grouped = _read_issues(out)
    all_ids = {issue_id for rows in grouped.values() for issue_id in rows}
    assert not any(issue_id.startswith("l1.video_") for issue_id in all_ids)
    assert not any(issue_id.startswith("l3.") for issue_id in all_ids)
    assert not (out / "video_quality.csv").exists()
    assert not (out / "av_sync.csv").exists()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Video metadata checks: `skipped`" in summary


def test_no_video_quality_skips_only_frame_quality(synthetic_root, tmp_path):
    out = tmp_path / "check_novq"
    assert main(
        [
            "check",
            "--dataset",
            str(synthetic_root),
            "--out",
            str(out),
            "--no-video-quality",
            "--full-report",
            "--no-progress",
        ]
    ) == 0
    assert not (out / "video_quality.csv").exists()
    # ffprobe metadata and PTS alignment are still produced.
    assert (out / "trajectory_scores.csv").exists()
    assert (out / "av_sync.csv").exists()


def test_output_inside_dataset_is_refused(synthetic_root):
    out = synthetic_root / "check_should_not_exist"
    assert main(
        ["check", "--dataset", str(synthetic_root), "--out", str(out), "--full-report", "--no-progress"]
    ) == 2
    assert not out.exists()
