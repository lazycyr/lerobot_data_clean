"""Write concise, VS Code-friendly check outputs: CSV + JSON + summary.md."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .av_sync import AVSyncRow
from .config import Config
from .dataset import EpisodeMeta, video_cameras
from .scoring import Issue, severity_rank, sort_issues, worst_severity
from .trajectory import TrajectoryEpisodeMetric, calibrate_trajectory
from .video_quality import VideoQualityRow


_EPISODE_FIELDS = [
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

_TRAJECTORY_FIELDS = [
    "episode_index",
    "computed",
    "max_smoothed_score",
    "max_score_frame",
    "max_abs_offset",
    "max_abs_frame",
    "worst_dim",
    "worst_dim_name",
    "segment_start",
    "segment_end",
    "flagged",
    "severity",
    "note",
]

_VIDEO_QUALITY_FIELDS = [
    "episode_index",
    "camera",
    "samples",
    "black_ratio",
    "dark_ratio",
    "bright_ratio",
    "freeze_ratio",
    "blur_low_ratio",
    "blur_min",
    "blur_median",
    "luma_min",
    "luma_median",
    "luma_max",
    "status",
    "issues",
]

_AV_SYNC_FIELDS = [
    "episode_index",
    "camera",
    "length",
    "video_frames",
    "action_fps",
    "video_fps",
    "action_start_s",
    "video_first_pts",
    "video_last_pts",
    "start_offset_s",
    "max_drift_s",
    "max_drift_frame",
    "pts_gap_max_dev_s",
    "status",
    "issue",
]


def _md(value: object) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _frame_text(issue: Issue, cfg: Config) -> str:
    if issue.frame_start is None or issue.frame_end is None:
        return "-"
    if issue.frame_start == issue.frame_end:
        return f"{issue.frame_start} (t={issue.frame_start / cfg.fps:.2f}s)"
    return (
        f"{issue.frame_start}..{issue.frame_end} "
        f"(t={issue.frame_start / cfg.fps:.2f}..{issue.frame_end / cfg.fps:.2f}s)"
    )


def write_episode_quality(
    out_dir: Path, issues: list[Issue], cfg: Config, *, include_json: bool = False
) -> tuple[Path, Path | None]:
    """Write only problematic episodes as CSV, optionally with a JSON copy.

    The concise default report is ``summary.md`` + ``episode_quality.csv``.
    JSON is only written with ``--full-report``.
    """
    episode_issues = [issue for issue in issues if issue.episode_index is not None]
    episode_issues = sort_issues(episode_issues)
    rows = []
    for issue in episode_issues:
        rows.append(
            {
                "episode_index": int(issue.episode_index),
                "issue_id": issue.issue_id,
                "severity": issue.severity,
                "frame_start": issue.frame_start,
                "frame_end": issue.frame_end,
                "time_start_s": (
                    round(issue.frame_start / cfg.fps, 6)
                    if issue.frame_start is not None
                    else None
                ),
                "time_end_s": (
                    round(issue.frame_end / cfg.fps, 6)
                    if issue.frame_end is not None
                    else None
                ),
                "event_count": int(issue.event_count),
                "detail": issue.detail,
                "action": issue.action,
            }
        )

    csv_path = out_dir / "episode_quality.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EPISODE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in _EPISODE_FIELDS})

    if not include_json:
        return csv_path, None
    json_path = out_dir / "episode_quality.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def write_trajectory_outputs(
    out_dir: Path,
    metrics: list[TrajectoryEpisodeMetric],
    cfg: Config,
) -> tuple[Path, Path, Path, dict]:
    """Write all-episode trajectory scores + calibration suggestion.

    These files intentionally include clean episodes: they are the data needed
    to choose a threshold for a new dataset, not the problem list.
    """
    rows = [asdict(metric) for metric in metrics]

    csv_path = out_dir / "trajectory_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TRAJECTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered = {}
            for key in _TRAJECTORY_FIELDS:
                value = row.get(key)
                if value is None:
                    rendered[key] = ""
                elif isinstance(value, bool):
                    rendered[key] = "true" if value else "false"
                else:
                    rendered[key] = value
            writer.writerow(rendered)

    json_path = out_dir / "trajectory_scores.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    calibration = calibrate_trajectory(metrics, cfg.trajectory_z_warn)
    calibration_path = out_dir / "trajectory_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, calibration_path, calibration


def write_video_quality_outputs(
    out_dir: Path,
    rows: list[VideoQualityRow],
    calibration: dict,
) -> tuple[Path, Path, Path]:
    """Write per-episode/camera frame-quality metrics and calibration."""
    payload = [asdict(row) for row in rows]

    csv_path = out_dir / "video_quality.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_VIDEO_QUALITY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in payload:
            rendered = {}
            for key in _VIDEO_QUALITY_FIELDS:
                value = row.get(key)
                rendered[key] = "" if value is None else value
            writer.writerow(rendered)

    json_path = out_dir / "video_quality.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    calibration_path = out_dir / "video_quality_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, json_path, calibration_path


def write_av_sync_outputs(out_dir: Path, rows: list[AVSyncRow]) -> tuple[Path, Path]:
    """Write all-episode/camera PTS alignment metrics."""
    payload = [asdict(row) for row in rows]
    csv_path = out_dir / "av_sync.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_AV_SYNC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in payload:
            rendered = {}
            for key in _AV_SYNC_FIELDS:
                value = row.get(key)
                rendered[key] = "" if value is None else value
            writer.writerow(rendered)
    json_path = out_dir / "av_sync.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _group_episode_issues(issues: list[Issue]) -> dict[int, list[Issue]]:
    grouped: dict[int, list[Issue]] = defaultdict(list)
    for issue in issues:
        if issue.episode_index is not None:
            grouped[int(issue.episode_index)].append(issue)
    return grouped


def _category_for(issue_id: str) -> str:
    if issue_id.startswith("l0."):
        return "完整性检查"
    if issue_id.startswith("l1."):
        return "时序对齐"
    if issue_id.startswith("l2."):
        return "轨迹异常"
    return "视频异常"


def _render_category(
    lines: list[str],
    title: str,
    issues: list[Issue],
    cfg: Config,
    *,
    video_quality_ran: bool | None = None,
    coverage_text: str | None = None,
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if coverage_text:
        lines.append(f"- 检查范围：{coverage_text}")
        lines.append("")
    actionable = [issue for issue in issues if issue.severity in ("fail", "warn")]
    info = [issue for issue in issues if issue.severity == "info"]
    if not actionable:
        lines.append("未检出异常。")
        if title == "视频异常" and video_quality_ran is False:
            lines.append("")
            lines.append("（视频帧质量检查未运行。）")
        if info:
            lines.append("")
            lines.append("备注（info）：")
            for issue in info:
                lines.append(f"- `{issue.issue_id}`: {issue.detail}")
        lines.append("")
        return

    lines.append("| Scope | Issue | Severity | Frames | Detail |")
    lines.append("|---|---|---|---|---|")
    ordered = sort_issues(actionable)
    dataset_rows = [issue for issue in ordered if issue.episode_index is None]
    episode_rows = [issue for issue in ordered if issue.episode_index is not None]
    for issue in dataset_rows:
        lines.append(
            f"| dataset | `{_md(issue.issue_id)}` | {_md(issue.severity)} | - | "
            f"{_md(issue.detail)} |"
        )
    for issue in episode_rows[: cfg.summary_max_episodes]:
        detail = issue.detail
        if len(detail) > 220:
            detail = detail[:217] + "..."
        lines.append(
            f"| {issue.episode_index} | `{_md(issue.issue_id)}` | {_md(issue.severity)} | "
            f"{_md(_frame_text(issue, cfg))} | {_md(detail)} |"
        )
    if len(episode_rows) > cfg.summary_max_episodes:
        lines.append("")
        lines.append(
            f"仅显示前 {cfg.summary_max_episodes} 个 episode 行；完整列表见 "
            "`episode_quality.csv`。"
        )
    if info:
        lines.append("")
        lines.append("备注（info）：")
        for issue in info:
            lines.append(f"- `{issue.issue_id}`: {issue.detail}")
    lines.append("")


def _render_calibration(
    lines: list[str],
    trajectory_scores_path: Path | None,
    trajectory_calibration: dict | None,
    video_quality_path: Path | None,
    video_quality_rows: list[VideoQualityRow] | None,
    video_quality_calibration: dict | None,
) -> None:
    if trajectory_scores_path is not None:
        lines.append("## Trajectory calibration")
        lines.append("")
        lines.append(f"- All-episode scores: `{trajectory_scores_path.name}`")
        if trajectory_calibration:
            computed = trajectory_calibration.get("computed_episodes", 0)
            current = trajectory_calibration.get("current_threshold")
            suggested = trajectory_calibration.get("suggested_threshold")
            gap = trajectory_calibration.get("largest_gap")
            lines.append(
                f"- Computed episodes: `{computed}`; current threshold: `{current}`"
            )
            if suggested is not None:
                extra = ""
                if gap:
                    extra = f" (largest gap {gap['low']} -> {gap['high']})"
                lines.append(
                    f"- Suggested threshold from this dataset: `{suggested}`{extra}"
                )
            lines.append(
                "- For a new dataset, inspect `trajectory_scores.csv` and choose a "
                "threshold from its own score distribution before reusing one."
            )
        lines.append("")

    if video_quality_path is not None:
        lines.append("## Video quality calibration")
        lines.append("")
        if video_quality_path is not None:
            lines.append(f"- Per-episode/camera metrics: `{video_quality_path.name}`")
        if video_quality_calibration and video_quality_calibration.get("enabled"):
            mode = (
                "full-frame"
                if video_quality_calibration.get("full_frame")
                else f"{video_quality_calibration.get('sample_fps')} fps"
            )
            lines.append(
                f"- Mode: `{mode}`, "
                f"`{video_quality_calibration.get('frame_width')}x"
                f"{video_quality_calibration.get('frame_height')}` grayscale"
            )
            for camera, stats in (video_quality_calibration.get("cameras") or {}).items():
                lines.append(
                    f"- `{camera}`: blur median `{stats.get('blur_p50')}`, "
                    f"threshold `{stats.get('blur_threshold')}`"
                )
        lines.append("")


def write_summary(
    out_dir: Path,
    root: Path,
    info: dict,
    episodes: list[EpisodeMeta],
    selected: list[int],
    episode_issues: list[Issue],
    dataset_issues: list[Issue],
    cfg: Config,
    *,
    elapsed_s: float,
    no_video: bool,
    scope_text: str,
    trajectory_scores_path: Path | None = None,
    trajectory_calibration: dict | None = None,
    video_quality_path: Path | None = None,
    video_quality_rows: list[VideoQualityRow] | None = None,
    video_quality_calibration: dict | None = None,
    av_sync_path: Path | None = None,
    av_sync_rows: list[AVSyncRow] | None = None,
    full_report: bool = False,
) -> Path:
    grouped = _group_episode_issues(episode_issues)
    selected_set = set(selected)
    issue_episodes = sorted(set(grouped) & selected_set)
    fail_episodes = sorted(
        ep for ep, rows in grouped.items() if ep in selected_set and any(i.severity == "fail" for i in rows)
    )
    warn_episodes = sorted(
        ep
        for ep, rows in grouped.items()
        if ep in selected_set
        and ep not in fail_episodes
        and any(i.severity == "warn" for i in rows)
    )
    info_episodes = sorted(
        ep
        for ep, rows in grouped.items()
        if ep in selected_set
        and ep not in fail_episodes
        and ep not in warn_episodes
        and any(i.severity == "info" for i in rows)
    )
    clean_count = len(selected_set) - len(issue_episodes)
    cameras = video_cameras(info)
    tasks = sorted({task for ep in episodes for task in ep.tasks})

    lines: list[str] = []
    lines.append("# dclean check summary")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Dataset: `{root}`")
    lines.append(
        f"- Codebase: `{info.get('codebase_version')}`, robot: `{info.get('robot_type', 'unknown')}`, "
        f"fps: `{info.get('fps')}`"
    )
    lines.append(
        f"- Episodes: `{len(episodes)}`, frames: `{info.get('total_frames')}`, "
        f"tasks: `{len(tasks)}`, cameras: `{', '.join(cameras) if cameras else 'none'}`"
    )
    lines.append(f"- Check scope: `{scope_text}`")
    lines.append(f"- Video metadata checks: `{'skipped' if no_video else 'enabled'}`")
    lines.append(
        f"- Report detail: `{'full' if full_report else 'main'}`"
        + ("" if full_report else " (use `--full-report` for all-episode tables and calibration JSON)")
    )
    lines.append(
        f"- Action/video PTS alignment: `{'skipped' if av_sync_rows is None else 'enabled'}`"
    )
    if av_sync_path is not None:
        lines.append(f"- PTS alignment table: `{av_sync_path.name}`")
    lines.append(f"- Elapsed: `{elapsed_s:.2f}s`")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"Checked **{len(selected_set)}** episode(s): **{clean_count}** clean, "
        f"**{len(warn_episodes)}** with warnings, **{len(fail_episodes)}** with failures"
        + (f", **{len(info_episodes)}** info-only." if info_episodes else ".")
    )
    lines.append("")
    if full_report:
        lines.append(
            "`episode_quality.csv` / `episode_quality.json` contain only problematic "
            "episodes; clean episodes are omitted."
        )
    else:
        lines.append(
            "`episode_quality.csv` contains only problematic episodes; "
            "clean episodes are omitted."
        )
    lines.append("")

    for title in ("完整性检查", "时序对齐", "轨迹异常", "视频异常"):
        category_issues = [
            issue
            for issue in (dataset_issues + episode_issues)
            if _category_for(issue.issue_id) == title
            and (issue.episode_index is None or issue.episode_index in selected_set)
        ]
        coverage_text = None
        if title == "时序对齐" and av_sync_rows is not None:
            selected_lengths = [
                ep.length for ep in episodes if ep.episode_index in selected_set
            ]
            if selected_lengths:
                length_text = (
                    f"{min(selected_lengths)} 帧"
                    if min(selected_lengths) == max(selected_lengths)
                    else f"{min(selected_lengths)}..{max(selected_lengths)} 帧"
                )
                coverage_text = (
                    f"{len(selected_set)} 集 × {len(cameras)} 相机；"
                    f"每集 {length_text}；"
                    f"{len(av_sync_rows)} 行 action/video PTS 对齐指标"
                )
        if title == "视频异常" and video_quality_rows is not None:
            selected_lengths = [
                ep.length for ep in episodes if ep.episode_index in selected_set
            ]
            if selected_lengths:
                coverage_text = (
                    f"{len(selected_set)} 集 × {len(cameras)} 相机；"
                    f"episode 帧 0..{max(selected_lengths) - 1}；"
                    f"每相机 {sum(selected_lengths)} 帧；"
                    f"{len(video_quality_rows)} 行 episode/camera 指标"
                )
        _render_category(
            lines,
            title,
            category_issues,
            cfg,
            video_quality_ran=(video_quality_rows is not None)
            if title == "视频异常"
            else None,
            coverage_text=coverage_text,
        )

    _render_calibration(
        lines,
        trajectory_scores_path,
        trajectory_calibration,
        video_quality_path,
        video_quality_rows,
        video_quality_calibration,
    )

    lines.append("## Next steps")
    lines.append("")
    if fail_episodes:
        lines.append(
            f"- Review the {len(fail_episodes)} failing episode(s) first: "
            f"{', '.join(str(e) for e in fail_episodes[:20])}"
            + (" ..." if len(fail_episodes) > 20 else "")
            + "."
        )
    if warn_episodes:
        lines.append(
            f"- Review the {len(warn_episodes)} warning episode(s) before deleting anything; "
            "warnings are candidates, not automatic removals."
        )
    if dataset_issues:
        lines.append(
            "- Dataset-level findings (e.g. codec declaration mismatch) are metadata notes; "
            "they do not by themselves make an episode unusable."
        )
    if not grouped and not dataset_issues:
        lines.append("- No issues found.")
    lines.append("")

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
