from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from dclean.cli import main
from dclean.dataset import (
    extract_episode_arrays,
    load_episodes,
    load_info,
    load_tasks,
    read_data_table,
    resolve_data_path,
)
from dclean.stats import column_to_numpy, compute_stats
from tests.fixtures import make_synthetic_dataset, snapshot_tree


DELETE = "6,9"
KEEP = [0, 1, 2, 3, 4, 5, 7, 8]


@pytest.fixture(scope="session")
def clean_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to build the synthetic video fixtures")
    root = tmp_path_factory.mktemp("clean_v30") / "dataset"
    return make_synthetic_dataset(root, inject_problems=False)


@pytest.fixture(scope="session")
def clean_snapshot(clean_root: Path) -> dict[str, tuple[int, int]]:
    return snapshot_tree(clean_root)


@pytest.fixture(scope="session")
def cleaned(clean_root: Path, clean_snapshot, tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("cleaned") / "dataset"
    check_csv = out.parent / "check_reasons.csv"
    with check_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "issue_id",
                "severity",
                "frame_start",
                "frame_end",
                "detail",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode_index": 6,
                "issue_id": "l2.trajectory_deviation",
                "severity": "fail",
                "frame_start": 10,
                "frame_end": 20,
                "detail": "synthetic reason",
            }
        )
    code = main(
        [
            "clean",
            "--dataset",
            str(clean_root),
            "--out",
            str(out),
            "--episodes",
            DELETE,
            "--check-report",
            str(check_csv),
            "--no-progress",
        ]
    )
    assert code == 0, "clean command failed"
    assert snapshot_tree(clean_root) == clean_snapshot
    return out


def test_output_metadata_and_data(cleaned, clean_root):
    info = load_info(cleaned)
    tasks = load_tasks(cleaned)
    episodes = load_episodes(cleaned, info)

    assert info["total_episodes"] == 8
    assert info["total_frames"] == 8 * 30
    assert info["total_tasks"] == 1
    assert info["splits"] == {"train": "0:8"}
    assert tasks == {0: "Part Sorting"}
    assert [ep.episode_index for ep in episodes] == list(range(8))
    assert all(ep.length == 30 for ep in episodes)

    for new, old in enumerate(KEEP):
        ep = episodes[new]
        table = read_data_table(resolve_data_path(cleaned, info, ep))
        arrays = extract_episode_arrays(table, ep)
        assert arrays.num_rows == 30
        assert np.array_equal(arrays.frame_index, np.arange(30))
        assert np.all(arrays.episode_index == new)
        assert np.array_equal(arrays.index, np.arange(new * 30, (new + 1) * 30))
        assert np.all(arrays.task_index == 0)
        # timestamp is episode-local and is not changed by cleaning.
        np.testing.assert_allclose(
            arrays.timestamp, np.arange(30) / 30.0, rtol=0, atol=1e-7
        )

    data_files = sorted((cleaned / "data" / "chunk-000").glob("file-*.parquet"))
    assert [path.name for path in data_files] == [f"file-{i:03d}.parquet" for i in range(8)]


def test_video_copy_and_reencode_branches(cleaned, clean_root):
    wrist = "observation.images.wrist_right"
    copied = cleaned / "videos" / wrist / "chunk-000" / "file-000.mp4"
    source_copy = clean_root / "videos" / wrist / "chunk-000" / "file-000.mp4"
    assert copied.is_file()
    assert hashlib.sha256(copied.read_bytes()).digest() == hashlib.sha256(
        source_copy.read_bytes()
    ).digest()

    # head_left file-000 contains deleted episodes and must be re-encoded.
    reencoded = cleaned / "videos" / "observation.images.head_left" / "chunk-000" / "file-000.mp4"
    source_reencoded = clean_root / "videos" / "observation.images.head_left" / "chunk-000" / "file-000.mp4"
    assert reencoded.is_file()
    assert reencoded.stat().st_size != source_reencoded.stat().st_size

    info = load_info(cleaned)
    episodes = load_episodes(cleaned, info)
    for ep in episodes:
        for camera, seg in ep.videos.items():
            assert 0 <= seg.from_timestamp < seg.to_timestamp
            assert seg.to_timestamp - seg.from_timestamp == pytest.approx(ep.length / 30.0)


def test_stats_are_exact_and_images_are_reused(cleaned):
    info = load_info(cleaned)
    ep_meta = pq.read_table(cleaned / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = {int(row["episode_index"]): row for row in ep_meta.to_pylist()}
    episodes = load_episodes(cleaned, info)

    for ep in episodes:
        table = pq.read_table(resolve_data_path(cleaned, info, ep))
        row = rows[ep.episode_index]
        for feature, dim in (("action", 20), ("observation.state", 48)):
            values = column_to_numpy(table.column(feature), dim)
            expected = compute_stats(values)
            for key in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                actual = np.asarray(row[f"stats/{feature}/{key}"], dtype=np.float64)
                np.testing.assert_allclose(actual, expected[key], rtol=1e-12, atol=1e-12)
        assert row["stats/index/min"] == [ep.dataset_from_index]
        assert row["stats/index/max"] == [ep.dataset_to_index - 1]
        assert row["stats/index/count"] == [ep.length]
        assert row["stats/episode_index/min"] == [ep.episode_index]
        assert row["stats/episode_index/max"] == [ep.episode_index]
        assert row["stats/task_index/min"] == [0]
        assert row["stats/task_index/max"] == [0]
        assert row["stats/task_index/count"] == [ep.length]

    stats = json.loads((cleaned / "meta" / "stats.json").read_text(encoding="utf-8"))
    total_frames = 8 * 30
    for feature in ("action", "observation.state", "timestamp", "frame_index",
                    "episode_index", "index", "task_index"):
        assert stats[feature]["count"] == [total_frames]
    assert len(stats["action"]["min"]) == 20
    assert len(stats["observation.state"]["min"]) == 48
    assert stats["index"]["min"] == [0]
    assert stats["index"]["max"] == [total_frames - 1]
    assert stats["episode_index"]["min"] == [0]
    assert stats["episode_index"]["max"] == [7]

    # Image stats: per-episode values are copied, global values are weighted.
    for camera in info["features"]:
        if not camera.startswith("observation.images."):
            continue
        assert len(stats[camera]["mean"]) == 3
        assert len(stats[camera]["mean"][0]) == 1
        assert len(stats[camera]["mean"][0][0]) == 1
        per_episode_count = 30 * 64 * 64
        assert stats[camera]["count"] == [8 * per_episode_count]
        expected_center = sum(0.4 + 0.01 * old for old in KEEP) / len(KEEP)
        np.testing.assert_allclose(stats[camera]["mean"], expected_center, rtol=1e-12)
        np.testing.assert_allclose(stats[camera]["min"], 0.0)
        np.testing.assert_allclose(stats[camera]["max"], 1.0)

    # The source image stats are unchanged on retained episodes (old ep 7 -> new 6).
    np.testing.assert_allclose(
        rows[6]["stats/observation.images.head_left/mean"], 0.47, rtol=1e-12
    )


def test_cleaning_reports(cleaned):
    report = (cleaned / "cleaning_report.md").read_text(encoding="utf-8")
    assert "数据清洗报告" in report
    assert "l2.trajectory_deviation" in report
    assert "复制" in report and "重编码" in report
    assert "通过" in report

    with (cleaned / "deleted_episodes.csv").open(newline="", encoding="utf-8") as handle:
        deleted = list(csv.DictReader(handle))
    assert [int(row["episode_index"]) for row in deleted] == [6, 9]
    assert "l2.trajectory_deviation" in deleted[0]["reasons"]
    assert deleted[1]["reasons"] == "manual selection (no check row)"

    with (cleaned / "index_mapping.csv").open(newline="", encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))
    assert [int(row["old_episode_index"]) for row in mapping] == KEEP
    assert [int(row["new_episode_index"]) for row in mapping] == list(range(8))
    assert mapping[6]["data_file"] == "data/chunk-000/file-006.parquet"


def test_output_passes_check(cleaned, tmp_path):
    check_out = tmp_path / "check_clean"
    code = main(
        [
            "check",
            "--dataset",
            str(cleaned),
            "--out",
            str(check_out),
            "--no-video-quality",
            "--full-report",
            "--no-progress",
        ]
    )
    assert code == 0
    rows = json.loads((check_out / "episode_quality.json").read_text(encoding="utf-8"))
    assert rows == []
    summary = (check_out / "summary.md").read_text(encoding="utf-8")
    assert "**8** clean" in summary
    assert "**0** with warnings" in summary
    assert "**0** with failures" in summary


def test_dry_run_writes_nothing(clean_root, clean_snapshot, tmp_path, capsys):
    out = tmp_path / "dry_run_dataset"
    code = main(
        [
            "clean",
            "--dataset",
            str(clean_root),
            "--out",
            str(out),
            "--episodes",
            DELETE,
            "--dry-run",
            "--no-progress",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert not out.exists()
    assert "dry-run" in captured.out
    assert "copy" in captured.out
    assert "reencode" in captured.out
    assert snapshot_tree(clean_root) == clean_snapshot


def test_invalid_clean_requests(clean_root, tmp_path):
    out = tmp_path / "bad"
    assert main(
        [
            "clean",
            "--dataset",
            str(clean_root),
            "--out",
            str(out),
            "--episodes",
            "0-9",
            "--no-progress",
        ]
    ) == 2
    assert not out.exists()

    inside = clean_root / "should_not_exist"
    assert main(
        [
            "clean",
            "--dataset",
            str(clean_root),
            "--out",
            str(inside),
            "--episodes",
            "6",
            "--no-progress",
        ]
    ) == 2
    assert not inside.exists()

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("x", encoding="utf-8")
    assert main(
        [
            "clean",
            "--dataset",
            str(clean_root),
            "--out",
            str(nonempty),
            "--episodes",
            "6",
            "--no-progress",
        ]
    ) == 2
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "x"
