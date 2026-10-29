"""Check orchestration: metadata -> L0 -> L1 -> L2 -> L3 -> report."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .checks import (
    check_action_state_residual,
    check_constant_channels,
    check_episode_structure,
    check_finite,
    check_inventory,
    check_joint_steps,
    check_metadata_totals,
    check_timestamps,
    global_constant_issue,
    global_dim_stats,
    step_baseline,
)
from .config import Config
from .dataset import (
    EpisodeArrays,
    EpisodeMeta,
    extract_episode_arrays,
    group_episodes_by_data_file,
    load_episodes,
    load_info,
    load_tasks,
    parse_episode_spec,
    read_data_table,
    resolve_video_path,
    video_cameras,
)
from .progress import Reporter
from .av_sync import AVSyncRow, check_av_sync
from .report import (
    write_av_sync_outputs,
    write_episode_quality,
    write_summary,
    write_trajectory_outputs,
    write_video_quality_outputs,
)
from .scoring import (
    Issue,
    dataset_issue,
    make_issue,
    merge_issues,
    suppress_covered_step_issues,
)
from .trajectory import check_trajectory_deviation
from .video import check_video
from .video_quality import VideoQualityRow, check_video_quality


@dataclass
class CheckResult:
    dataset_issues: list[Issue]
    episode_issues: list[Issue]
    selected: list[int]
    elapsed_s: float
    summary_path: Path
    quality_csv_path: Path
    quality_json_path: Path | None
    trajectory_scores_path: Path | None = None
    trajectory_scores_json_path: Path | None = None
    trajectory_calibration_path: Path | None = None
    video_quality_path: Path | None = None
    video_quality_json_path: Path | None = None
    video_quality_calibration_path: Path | None = None
    av_sync_path: Path | None = None
    av_sync_json_path: Path | None = None


def run_check(
    dataset_root: Path,
    out_dir: Path,
    episode_spec: str | None,
    cfg: Config,
    *,
    no_video: bool = False,
    no_video_quality: bool = False,
    full_report: bool = False,
    reporter: Reporter | None = None,
) -> CheckResult:
    started = time.monotonic()
    reporter = reporter or Reporter(enabled=True)
    dataset_root = dataset_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_issues: list[Issue] = []
    episode_issues: list[Issue] = []

    # -- stage 1: metadata -------------------------------------------------
    reporter.stage("加载元数据")
    info = load_info(dataset_root)
    tasks = load_tasks(dataset_root)
    episodes = load_episodes(dataset_root, info)
    selected = parse_episode_spec(episode_spec, int(info["total_episodes"]))
    by_index = {ep.episode_index: ep for ep in episodes}
    missing_selected = [idx for idx in selected if idx not in by_index]
    if missing_selected:
        raise ValueError(
            f"selected episode(s) not present in metadata: {missing_selected[:8]}"
        )
    selected_eps = [by_index[idx] for idx in selected]
    reporter.info(
        f"  dataset: {info.get('codebase_version')} / {info.get('robot_type', 'unknown')}, "
        f"{len(episodes)} episodes, {info.get('total_frames')} frames, "
        f"{len(video_cameras(info))} cameras"
    )

    # -- stage 2: L0 metadata + inventory ---------------------------------
    reporter.stage("L0 元数据/目录完整性")
    dataset_issues.extend(check_metadata_totals(dataset_root, info, tasks, episodes))
    inventory_issues = check_inventory(
        dataset_root, info, episodes, selected_indices=set(selected)
    )
    dataset_issues.extend(i for i in inventory_issues if i.episode_index is None)
    episode_issues.extend(i for i in inventory_issues if i.episode_index is not None)
    reporter.info(
        f"  -> dataset issues: {len(dataset_issues)}, episode issues: {len(episode_issues)}"
    )

    # -- stage 3: read data + L0 per-episode + L1 -------------------------
    reporter.stage("读取数据 + L1 时间戳/视频对齐")
    arrays_by_episode: dict[int, EpisodeArrays] = {}
    groups = group_episodes_by_data_file(dataset_root, info, selected_eps)
    action_dim = int((info.get("features", {}).get("action", {}).get("shape") or [20])[0])
    state_dim = int(
        (info.get("features", {}).get("observation.state", {}).get("shape") or [48])[0]
    )
    bar = reporter.bar(len(groups), "读取数据")
    for path, group in sorted(groups.items(), key=lambda item: str(item[0])):
        try:
            table = read_data_table(path)
        except Exception as exc:  # noqa: BLE001 - per-file failure, keep checking
            for ep in group:
                episode_issues.append(
                    make_issue(
                        "l0.data_file_unreadable",
                        "fail",
                        ep.episode_index,
                        0,
                        max(0, ep.length - 1),
                        1,
                        f"cannot read {path}: {exc}",
                    )
                )
            bar.update()
            continue
        for ep in group:
            try:
                arrays = extract_episode_arrays(
                    table, ep, action_dim=action_dim, state_dim=state_dim
                )
            except Exception as exc:  # noqa: BLE001
                episode_issues.append(
                    make_issue(
                        "l0.data_row_count",
                        "fail",
                        ep.episode_index,
                        0,
                        max(0, ep.length - 1),
                        1,
                        f"cannot extract episode rows: {exc}",
                    )
                )
                continue
            arrays_by_episode[ep.episode_index] = arrays
            episode_issues.extend(check_episode_structure(ep, arrays, tasks))
            episode_issues.extend(check_timestamps(ep, arrays, cfg))
        bar.update()
    bar.close()
    reporter.info(
        f"  -> loaded {len(arrays_by_episode)}/{len(selected_eps)} episodes, "
        f"episode issues: {len(episode_issues)}"
    )

    # -- stage 4: L2 numeric (dims 0..19 only) ----------------------------
    reporter.stage("L2 数值检查 (dims 0-19)")
    action_std, action_frac, action_global_const = global_dim_stats(
        arrays_by_episode, "action", cfg
    )
    state_std, state_frac, state_global_const = global_dim_stats(
        arrays_by_episode, "state", cfg
    )
    for issue in (
        global_constant_issue(info, "action", action_global_const),
        global_constant_issue(info, "state", state_global_const),
    ):
        if issue is not None:
            dataset_issues.append(issue)

    lo, hi = cfg.joint_dims
    action_base = step_baseline(arrays_by_episode, "action", lo, hi, cfg)
    state_base = step_baseline(arrays_by_episode, "state", lo, hi, cfg)

    bar = reporter.bar(len(selected_eps), "L2 数值")
    for ep in selected_eps:
        arrays = arrays_by_episode.get(ep.episode_index)
        if arrays is not None:
            for issue in (
                check_finite(ep, arrays),
                check_constant_channels(
                    ep,
                    arrays,
                    info,
                    action_std,
                    action_global_const,
                    state_std,
                    state_global_const,
                    cfg,
                ),
                check_joint_steps(ep, arrays, info, action_base, state_base, cfg),
                check_action_state_residual(ep, arrays, cfg),
            ):
                if issue is not None:
                    episode_issues.append(issue)
        bar.update()
    bar.close()

    trajectory_result = check_trajectory_deviation(arrays_by_episode, info, cfg)
    trajectory_metrics = trajectory_result.metrics
    episode_issues.extend(trajectory_result.issues)
    episode_issues = suppress_covered_step_issues(episode_issues)
    if trajectory_result.dataset_issue is not None:
        dataset_issues.append(trajectory_result.dataset_issue)
    reporter.info(
        f"  -> episode issues: {len(episode_issues)} "
        f"(trajectory: {len(trajectory_result.issues)})"
    )

    # -- stage 5: L3 video metadata + action/video PTS alignment ----------
    av_sync_rows: list[AVSyncRow] | None = None
    if no_video:
        reporter.stage("L3 视频元数据 + PTS 对齐（已跳过）")
    else:
        reporter.stage("L3 视频元数据")
        video_paths: set[Path] = set()
        for ep in selected_eps:
            for camera in video_cameras(info):
                if camera in ep.videos:
                    path = resolve_video_path(dataset_root, info, ep, camera)
                    if path.is_file() and path.stat().st_size > 0:
                        video_paths.add(path)
        bar = reporter.bar(len(video_paths), "视频元数据")
        video_dataset_issues, video_episode_issues = check_video(
            dataset_root,
            info,
            selected_eps,
            all_episodes=episodes,
            arrays_by_episode=arrays_by_episode,
            cfg=cfg,
            progress=bar,
        )
        bar.close()
        dataset_issues.extend(video_dataset_issues)
        episode_issues.extend(video_episode_issues)
        if cfg.av_sync_enabled:
            bar = reporter.bar(len(video_paths), "PTS 对齐")
            (
                sync_dataset_issues,
                sync_episode_issues,
                av_sync_rows,
            ) = check_av_sync(
                dataset_root,
                info,
                selected_eps,
                arrays_by_episode,
                cfg,
                progress=bar,
            )
            bar.close()
            dataset_issues.extend(sync_dataset_issues)
            episode_issues.extend(sync_episode_issues)
            mismatches = sum(1 for row in av_sync_rows if row.status != "ok")
            reporter.info(
                f"  -> {len(video_paths)} video file(s), PTS alignment rows: "
                f"{len(av_sync_rows)}, mismatches: {mismatches}"
            )
        reporter.info(
            f"  -> dataset issues: {len(dataset_issues)}, episode issues: {len(episode_issues)}"
        )

    # -- stage 6: L4 video frame quality ---------------------------------
    video_quality_rows: list[VideoQualityRow] | None = None
    video_quality_calibration: dict | None = None
    if no_video or no_video_quality or not cfg.video_quality_enabled:
        reporter.stage("L4 视频帧质量（已跳过）")
    else:
        reporter.stage("L4 视频帧质量")
        quality_paths: set[Path] = set()
        for ep in selected_eps:
            for camera in video_cameras(info):
                if camera in ep.videos:
                    path = resolve_video_path(dataset_root, info, ep, camera)
                    if path.is_file() and path.stat().st_size > 0:
                        quality_paths.add(path)
        bar = reporter.bar(len(quality_paths), "视频帧质量")
        (
            quality_dataset_issues,
            quality_episode_issues,
            video_quality_rows,
            video_quality_calibration,
        ) = check_video_quality(
            dataset_root,
            info,
            selected_eps,
            cfg,
            progress=bar,
        )
        bar.close()
        dataset_issues.extend(quality_dataset_issues)
        episode_issues.extend(quality_episode_issues)
        flagged = sum(1 for row in video_quality_rows if row.status != "ok")
        reporter.info(
            f"  -> {len(video_quality_rows)} episode/camera rows, {flagged} flagged"
        )

    # -- stage 7: report --------------------------------------------------
    reporter.stage("写报告")
    merged = merge_issues(episode_issues)
    quality_csv_path, quality_json_path = write_episode_quality(
        out_dir, merged, cfg, include_json=full_report
    )

    trajectory_scores_path = None
    trajectory_scores_json_path = None
    trajectory_calibration_path = None
    trajectory_calibration = None
    if full_report and trajectory_metrics:
        (
            trajectory_scores_path,
            trajectory_scores_json_path,
            trajectory_calibration_path,
            trajectory_calibration,
        ) = write_trajectory_outputs(out_dir, trajectory_metrics, cfg)

    video_quality_path = None
    video_quality_json_path = None
    video_quality_calibration_path = None
    if full_report and video_quality_rows is not None:
        (
            video_quality_path,
            video_quality_json_path,
            video_quality_calibration_path,
        ) = write_video_quality_outputs(
            out_dir, video_quality_rows, video_quality_calibration or {}
        )

    av_sync_path = None
    av_sync_json_path = None
    if full_report and av_sync_rows is not None:
        av_sync_path, av_sync_json_path = write_av_sync_outputs(out_dir, av_sync_rows)

    scope_text = (
        "all episodes"
        if len(selected) == len(episodes)
        else f"episodes {selected[0]}..{selected[-1]} ({len(selected)} selected)"
    )
    elapsed = time.monotonic() - started
    summary_path = write_summary(
        out_dir,
        dataset_root,
        info,
        episodes,
        selected,
        merged,
        dataset_issues,
        cfg,
        elapsed_s=elapsed,
        no_video=no_video,
        scope_text=scope_text,
        trajectory_scores_path=trajectory_scores_path,
        trajectory_calibration=trajectory_calibration,
        video_quality_path=video_quality_path,
        video_quality_rows=video_quality_rows,
        video_quality_calibration=video_quality_calibration,
        av_sync_path=av_sync_path,
        av_sync_rows=av_sync_rows,
        full_report=full_report,
    )
    return CheckResult(
        dataset_issues=dataset_issues,
        episode_issues=merged,
        selected=selected,
        elapsed_s=elapsed,
        summary_path=summary_path,
        quality_csv_path=quality_csv_path,
        quality_json_path=quality_json_path,
        trajectory_scores_path=trajectory_scores_path,
        trajectory_scores_json_path=trajectory_scores_json_path,
        trajectory_calibration_path=trajectory_calibration_path,
        video_quality_path=video_quality_path,
        video_quality_json_path=video_quality_json_path,
        video_quality_calibration_path=video_quality_calibration_path,
        av_sync_path=av_sync_path,
        av_sync_json_path=av_sync_json_path,
    )
