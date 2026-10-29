"""Materialize a cleaned LeRobot v3.0 dataset copy.

The source dataset is never modified. Cleaning follows LeRobot's
``delete_episodes`` flow (filter data, re-index episodes, rewrite video
segments, rebuild metadata/stats), with three deliberate fixes:

* data files are compacted to one file per retained episode;
* per-episode ``stats/`` for re-indexed columns are recomputed instead of
  copied stale from the source;
* the declared video codec is kept in sync with the actual output codec.
"""

from __future__ import annotations

import csv
import glob
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .checks import check_episode_structure, check_inventory, check_metadata_totals
from .dataset import (
    EpisodeMeta,
    extract_episode_arrays,
    load_episodes,
    load_info,
    load_tasks,
    parse_episode_spec,
    read_data_table,
    resolve_data_path,
    resolve_video_path,
    video_cameras,
)
from .progress import Reporter
from .stats import (
    STAT_KEYS,
    aggregate_image_stats,
    column_to_numpy,
    compute_stats,
    image_stats_from_row,
    stats_to_json,
)
from .video import VideoProbe, probe_video


REPORT_FILES = ("cleaning_report.md", "deleted_episodes.csv", "index_mapping.csv")


@dataclass(frozen=True)
class DeletedEpisode:
    episode_index: int
    length: int
    data_file: str
    video_files: str
    reasons: str


@dataclass(frozen=True)
class VideoJob:
    camera: str
    source: Path
    output_rel: Path
    chunk_index: int
    file_index: int
    mode: str  # "copy" or "reencode"
    kept_old: tuple[int, ...]
    segments: dict[int, tuple[float, float]]
    start_frames: dict[int, int]
    expected_frames: int
    reason: str
    pix_fmt: str
    source_probe: VideoProbe


@dataclass
class CleanPlan:
    source: Path
    info: dict
    tasks: dict[int, str]
    episodes: list[EpisodeMeta]
    deleted: list[int]
    keep: list[int]
    old_to_new: dict[int, int]
    new_tasks: dict[str, int]
    deleted_rows: list[DeletedEpisode]
    video_jobs: list[VideoJob]
    video_by_episode_camera: dict[tuple[int, str], VideoJob]
    skipped_video_files: list[str]
    warnings: list[str]
    lowdim_features: list[str]
    target_codec: str
    vcodec: str


@dataclass
class CleanResult:
    out: Path
    deleted_count: int
    kept_count: int
    total_frames: int
    report_path: Path
    deleted_csv: Path
    mapping_csv: Path
    verification_issues: list[str]
    warnings: list[str]


@dataclass
class _DataRewrite:
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]]
    values_by_feature: dict[str, list[np.ndarray]]
    data_refs: dict[int, tuple[int, int, int, int]]


def _codec_for_encoder(vcodec: str) -> str:
    normalized = vcodec.strip().lower()
    aliases = {
        "libx264": "h264",
        "h264": "h264",
        "libsvtav1": "av1",
        "av1": "av1",
    }
    if normalized not in aliases:
        raise ValueError(
            f"unsupported --vcodec {vcodec!r}; use libx264 (h264) or libsvtav1 (av1)"
        )
    return aliases[normalized]


def _encoder_for_vcodec(vcodec: str) -> str:
    normalized = vcodec.strip().lower()
    if normalized in {"h264", "libx264"}:
        return "libx264"
    if normalized in {"av1", "libsvtav1"}:
        return "libsvtav1"
    raise ValueError(
        f"unsupported --vcodec {vcodec!r}; use libx264 (h264) or libsvtav1 (av1)"
    )


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _read_check_reasons(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"--check-report does not exist: {path}")
    grouped: dict[int, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "episode_index" not in (reader.fieldnames or []):
            raise ValueError(f"--check-report has no episode_index column: {path}")
        for row in reader:
            try:
                episode = int(row["episode_index"])
            except (TypeError, ValueError):
                continue
            issue = row.get("issue_id") or "issue"
            severity = row.get("severity") or ""
            start = row.get("frame_start") or ""
            end = row.get("frame_end") or ""
            detail = (row.get("detail") or "").replace("\n", " ")
            if len(detail) > 220:
                detail = detail[:217] + "..."
            frames = f" frames {start}..{end}" if start and end else ""
            grouped[episode].append(
                f"{issue}[{severity}]{frames}: {detail}" if detail else f"{issue}[{severity}]{frames}"
            )
    return {episode: " | ".join(parts) for episode, parts in grouped.items()}


def _load_episode_table(root: Path) -> pa.Table:
    paths = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    tables: list[pa.Table] = []
    schema = None
    for path in paths:
        table = pq.read_table(path)
        if schema is None:
            schema = table.schema
        elif table.schema != schema:
            raise ValueError(f"episode metadata schema differs between shards: {path}")
        tables.append(table)
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _source_data_columns(source: Path, info: dict, episodes: list[EpisodeMeta]) -> set[str]:
    for episode in episodes:
        path = resolve_data_path(source, info, episode)
        if path.is_file():
            return set(pq.read_schema(path).names)
    raise FileNotFoundError("no source data file found for the first episodes")


def _mapping_runs(keep: list[int], old_to_new: dict[int, int]) -> list[tuple[int, int, int, int]]:
    """Return contiguous ``(old_start, old_end, new_start, new_end)`` runs."""
    if not keep:
        return []
    runs: list[tuple[int, int, int, int]] = []
    old_start = old_end = keep[0]
    new_start = new_end = old_to_new[keep[0]]
    for old in keep[1:]:
        new = old_to_new[old]
        if old == old_end + 1 and new == new_end + 1:
            old_end, new_end = old, new
        else:
            runs.append((old_start, old_end, new_start, new_end))
            old_start = old_end = old
            new_start = new_end = new
    runs.append((old_start, old_end, new_start, new_end))
    return runs


def _source_segments_safe(
    group: list[EpisodeMeta],
    camera: str,
    probe: VideoProbe,
    fps: float,
) -> tuple[bool, str]:
    if probe.nb_frames is None:
        return False, "source frame count unknown"
    ordered = sorted(group, key=lambda ep: ep.videos[camera].from_timestamp)
    previous_to = None
    for ep in ordered:
        seg = ep.videos[camera]
        if abs((seg.to_timestamp - seg.from_timestamp) * fps - ep.length) > 1.0:
            return False, f"ep{ep.episode_index} segment duration != length"
        start = int(round(seg.from_timestamp * fps))
        end = start + ep.length
        if start < 0 or end > probe.nb_frames + 1:
            return False, f"ep{ep.episode_index} segment outside source frames"
        if previous_to is not None and abs(previous_to - seg.from_timestamp) > 1e-6:
            return False, "adjacent episode segments are not contiguous"
        previous_to = seg.to_timestamp
    return True, ""


def _declared_video_spec(info: dict, camera: str) -> tuple[int | None, int | None]:
    feature = info.get("features", {}).get(camera, {})
    meta = feature.get("info", {}) or {}
    shape = feature.get("shape") or []
    width = meta.get("video.width", shape[1] if len(shape) > 1 else None)
    height = meta.get("video.height", shape[0] if len(shape) > 0 else None)
    return (
        int(width) if width is not None else None,
        int(height) if height is not None else None,
    )


def _plan_video_jobs(
    source: Path,
    info: dict,
    episodes: list[EpisodeMeta],
    keep: list[int],
    target_codec: str,
) -> tuple[list[VideoJob], dict[tuple[int, str], VideoJob], list[str], list[str]]:
    fps = float(info["fps"])
    keep_set = set(keep)
    cameras = video_cameras(info)
    groups: dict[tuple[str, Path], list[EpisodeMeta]] = {}
    missing: list[str] = []
    for ep in episodes:
        for camera in cameras:
            seg = ep.videos.get(camera)
            if seg is None:
                if ep.episode_index in keep_set:
                    raise ValueError(f"episode {ep.episode_index} has no {camera} video metadata")
                continue
            path = resolve_video_path(source, info, ep, camera)
            if not _nonempty(path):
                if ep.episode_index in keep_set:
                    missing.append(
                        f"ep{ep.episode_index} {camera}: missing source video {path}"
                    )
                continue
            groups.setdefault((camera, path), []).append(ep)
    if missing:
        raise ValueError("cannot clean; retained episodes reference missing videos:\n" + "\n".join(missing))

    jobs: list[VideoJob] = []
    by_episode: dict[tuple[int, str], VideoJob] = {}
    skipped: list[str] = []
    warnings: list[str] = []
    for (camera, path), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        kept = [ep for ep in group if ep.episode_index in keep_set]
        if not kept:
            skipped.append(f"{camera}: {path.relative_to(source)} (all episodes deleted)")
            continue
        probe = probe_video(path)
        if probe.error is not None:
            raise ValueError(f"ffprobe failed for retained video {path}: {probe.error}")
        if probe.fps is not None and abs(probe.fps - fps) > 1e-3:
            raise ValueError(
                f"{path}: source fps {probe.fps:.6f} != info.fps {fps:.6f}; "
                "cleaning does not resample video"
            )
        declared_w, declared_h = _declared_video_spec(info, camera)
        if (
            declared_w is not None
            and probe.width is not None
            and declared_w != probe.width
        ) or (
            declared_h is not None
            and probe.height is not None
            and declared_h != probe.height
        ):
            warnings.append(
                f"{camera} {path.name}: source {probe.width}x{probe.height}, "
                f"info declares {declared_w}x{declared_h}; cleaning preserves source resolution"
            )
        expected_frames = sum(ep.length for ep in kept)
        all_kept = len(kept) == len(group)
        segments_safe, segments_reason = _source_segments_safe(group, camera, probe, fps)
        codec_matches = (probe.codec or "").lower() == target_codec
        mode = "copy"
        if not all_kept:
            mode = "reencode"
            reason = f"{len(kept)}/{len(group)} episodes retained"
        elif not codec_matches:
            mode = "reencode"
            reason = f"codec {probe.codec or 'unknown'} -> {target_codec}"
        elif not segments_safe:
            mode = "reencode"
            reason = f"source segments not copy-safe: {segments_reason}"
        else:
            reason = "all episodes retained; codec matches"

        ordered_kept = sorted(kept, key=lambda ep: ep.videos[camera].from_timestamp)
        segments: dict[int, tuple[float, float]] = {}
        start_frames: dict[int, int] = {}
        if mode == "copy":
            for ep in ordered_kept:
                seg = ep.videos[camera]
                segments[ep.episode_index] = (seg.from_timestamp, seg.to_timestamp)
        else:
            cumulative = 0
            for ep in ordered_kept:
                seg = ep.videos[camera]
                start = int(round(seg.from_timestamp * fps))
                end = start + ep.length
                if start < 0:
                    raise ValueError(f"ep{ep.episode_index} {camera}: negative start frame {start}")
                if probe.nb_frames is not None and end > probe.nb_frames + 1:
                    raise ValueError(
                        f"ep{ep.episode_index} {camera}: frames {start}..{end - 1} "
                        f"exceed source nb_frames={probe.nb_frames}"
                    )
                start_frames[ep.episode_index] = start
                segments[ep.episode_index] = (
                    cumulative / fps,
                    (cumulative + ep.length) / fps,
                )
                cumulative += ep.length
            if cumulative != expected_frames:
                raise AssertionError("internal frame accounting error")
        assert len(segments) == len(kept)
        output_rel = path.relative_to(source)
        job = VideoJob(
            camera=camera,
            source=path,
            output_rel=output_rel,
            chunk_index=kept[0].videos[camera].chunk_index,
            file_index=kept[0].videos[camera].file_index,
            mode=mode,
            kept_old=tuple(ep.episode_index for ep in ordered_kept),
            segments=segments,
            start_frames=start_frames,
            expected_frames=expected_frames,
            reason=reason,
            pix_fmt=probe.pix_fmt or "yuv420p",
            source_probe=probe,
        )
        jobs.append(job)
        for ep in kept:
            key = (ep.episode_index, camera)
            if key in by_episode:
                raise ValueError(f"duplicate video mapping for ep{ep.episode_index} {camera}")
            by_episode[key] = job
    if not jobs:
        raise ValueError("no retained video files to write")
    return jobs, by_episode, skipped, warnings


def plan_clean(
    source: Path,
    deleted_spec: str,
    *,
    check_report: Path | None = None,
    vcodec: str = "libx264",
) -> CleanPlan:
    source = source.resolve()
    info = load_info(source)
    tasks = load_tasks(source)
    episodes = load_episodes(source, info)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(
            f"episode metadata has {len(episodes)} episodes, "
            f"info.json declares {info['total_episodes']}"
        )
    total = len(episodes)
    deleted = parse_episode_spec(deleted_spec, total)
    deleted_set = set(deleted)
    keep = [episode for episode in range(total) if episode not in deleted_set]
    if not keep:
        raise ValueError("refusing to delete every episode")
    old_to_new = {old: new for new, old in enumerate(keep)}
    target_codec = _codec_for_encoder(vcodec)
    encoder = _encoder_for_vcodec(vcodec)

    data_columns = _source_data_columns(source, info, episodes)
    lowdim_features = [
        name
        for name, spec in info.get("features", {}).items()
        if spec.get("dtype") != "video" and name in data_columns
    ]
    required = {"action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    missing = required - set(lowdim_features)
    if missing:
        raise ValueError(f"source data is missing required feature columns: {sorted(missing)}")
    for old in keep:
        data_path = resolve_data_path(source, info, episodes[old])
        if not _nonempty(data_path):
            raise ValueError(f"ep{old}: missing source data file {data_path}")

    used_task_names: set[str] = set()
    by_index = {ep.episode_index: ep for ep in episodes}
    for old in keep:
        used_task_names.update(by_index[old].tasks)
    new_tasks = {name: index for index, name in enumerate(sorted(used_task_names))}
    if not new_tasks:
        raise ValueError("retained episodes have no task names")

    check_reasons = _read_check_reasons(check_report)
    deleted_rows: list[DeletedEpisode] = []
    for old in deleted:
        ep = by_index[old]
        video_refs = "; ".join(
            f"{camera}={resolve_video_path(source, info, ep, camera).relative_to(source)}"
            for camera in video_cameras(info)
        )
        reason = check_reasons.get(old) or "manual selection (no check row)"
        deleted_rows.append(
            DeletedEpisode(
                episode_index=old,
                length=ep.length,
                data_file=str(resolve_data_path(source, info, ep).relative_to(source)),
                video_files=video_refs,
                reasons=reason,
            )
        )

    video_jobs, by_episode_camera, skipped, warnings = _plan_video_jobs(
        source, info, episodes, keep, target_codec
    )
    for old in keep:
        for camera in video_cameras(info):
            if (old, camera) not in by_episode_camera:
                raise ValueError(f"no output video planned for ep{old} {camera}")

    return CleanPlan(
        source=source,
        info=info,
        tasks=tasks,
        episodes=episodes,
        deleted=deleted,
        keep=keep,
        old_to_new=old_to_new,
        new_tasks=new_tasks,
        deleted_rows=deleted_rows,
        video_jobs=video_jobs,
        video_by_episode_camera=by_episode_camera,
        skipped_video_files=skipped,
        warnings=warnings,
        lowdim_features=lowdim_features,
        target_codec=target_codec,
        vcodec=encoder,
    )


def validate_clean_paths(source: Path, out: Path) -> tuple[Path, Path]:
    source = source.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise ValueError(f"not a LeRobot dataset (missing meta/info.json): {source}")
    out = out.resolve()
    if out == source or source in out.parents:
        raise ValueError("--out must not be inside --dataset")
    if out in source.parents:
        raise ValueError("--out must not be a parent directory of --dataset")
    return source, out


def prepare_clean_output(out: Path, overwrite: bool, cameras: list[str]) -> None:
    if out.exists():
        if not out.is_dir():
            raise ValueError(f"--out exists and is not a directory: {out}")
        if any(out.iterdir()):
            if not overwrite:
                raise ValueError(f"output directory is not empty (use --overwrite): {out}")
            for name in ("meta", "data", "videos"):
                target = out / name
                if target.exists():
                    shutil.rmtree(target)
            for name in REPORT_FILES:
                target = out / name
                if target.is_file():
                    target.unlink()
    else:
        out.mkdir(parents=True, exist_ok=True)

    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        (out / "videos" / camera).mkdir(parents=True, exist_ok=True)


def _episode_slice(
    table: pa.Table,
    episode: EpisodeMeta,
    source_path: Path,
) -> pa.Table:
    if "episode_index" not in table.column_names:
        raise ValueError(f"{source_path} has no episode_index column")
    episode_col = np.asarray(
        table.column("episode_index").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    sub = table.filter(pa.array(episode_col == episode.episode_index))
    if sub.num_rows != episode.length:
        raise ValueError(
            f"ep{episode.episode_index}: {source_path} has {sub.num_rows} rows, "
            f"metadata declares length={episode.length}"
        )
    frame_index = np.asarray(
        sub.column("frame_index").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    if not np.array_equal(frame_index, np.arange(episode.length, dtype=np.int64)):
        raise ValueError(
            f"ep{episode.episode_index}: frame_index is not 0..{episode.length - 1} "
            f"in row order in {source_path}"
        )
    return sub


def _reindex_data_table(
    sub: pa.Table,
    *,
    new_episode_index: int,
    global_start: int,
    old_task_indices: np.ndarray,
    task_mapping: dict[int, int],
) -> pa.Table:
    table = sub
    for name, values in (
        ("episode_index", np.full(sub.num_rows, new_episode_index, dtype=np.int64)),
        ("index", global_start + np.arange(sub.num_rows, dtype=np.int64)),
        (
            "task_index",
            np.asarray([task_mapping[int(value)] for value in old_task_indices], dtype=np.int64),
        ),
    ):
        index = table.schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"source data has no {name!r} column")
        field = table.schema.field(index)
        table = table.set_column(index, name, pa.array(values.tolist(), type=field.type))
    return table


def _feature_values(table: pa.Table, info: dict, feature: str) -> np.ndarray:
    spec = info.get("features", {}).get(feature)
    if spec is None:
        raise ValueError(f"info.json has no feature {feature!r}")
    shape = spec.get("shape") or []
    dim = int(shape[0]) if len(shape) == 1 else 1
    if dim <= 0:
        raise ValueError(f"feature {feature!r} has invalid shape {shape}")
    return column_to_numpy(table.column(feature), dim)


def _write_data(
    plan: CleanPlan,
    out: Path,
    reporter: Reporter,
) -> _DataRewrite:
    info = plan.info
    chunks_size = int(info.get("chunks_size", 1000))
    group: dict[Path, list[EpisodeMeta]] = defaultdict(list)
    for old in plan.keep:
        ep = plan.episodes[old]
        group[resolve_data_path(plan.source, info, ep)].append(ep)

    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    values_by_feature: dict[str, list[np.ndarray]] = defaultdict(list)
    data_refs: dict[int, tuple[int, int, int, int]] = {}
    old_to_name = plan.tasks
    name_to_new = plan.new_tasks
    old_to_new_task = {
        old: name_to_new[name]
        for old, name in old_to_name.items()
        if name in name_to_new
    }

    # Global offsets follow the new episode order, not the source file order.
    prefix: dict[int, int] = {}
    offset = 0
    for old in plan.keep:
        prefix[old] = offset
        offset += plan.episodes[old].length

    progress = reporter.bar(len(plan.keep), "data episodes")
    try:
        for source_path in sorted(group, key=str):
            if not source_path.is_file():
                raise FileNotFoundError(f"missing source data file {source_path}")
            table = pq.read_table(source_path)
            for ep in sorted(group[source_path], key=lambda item: item.episode_index):
                sub = _episode_slice(table, ep, source_path)
                old_task_indices = np.asarray(
                    sub.column("task_index").to_numpy(zero_copy_only=False), dtype=np.int64
                )
                unknown = sorted(
                    {int(value) for value in old_task_indices if int(value) not in old_to_new_task}
                )
                if unknown:
                    raise ValueError(
                        f"ep{ep.episode_index}: task_index {unknown} has no retained task mapping"
                    )
                new_index = plan.old_to_new[ep.episode_index]
                global_start = prefix[ep.episode_index]
                new_table = _reindex_data_table(
                    sub,
                    new_episode_index=new_index,
                    global_start=global_start,
                    old_task_indices=old_task_indices,
                    task_mapping=old_to_new_task,
                )
                chunk_index = new_index // chunks_size
                file_index = new_index % chunks_size
                rel = info["data_path"].format(
                    chunk_index=chunk_index, file_index=file_index
                )
                data_path = out / rel
                data_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(new_table, data_path)

                stats: dict[str, dict[str, np.ndarray]] = {}
                for feature in plan.lowdim_features:
                    values = _feature_values(new_table, info, feature)
                    try:
                        feature_stats = compute_stats(values)
                    except ValueError as exc:
                        raise ValueError(
                            f"ep{ep.episode_index} feature {feature!r}: {exc}"
                        ) from exc
                    stats[feature] = feature_stats
                    values_by_feature[feature].append(values)
                episode_stats[ep.episode_index] = stats
                data_refs[ep.episode_index] = (
                    chunk_index,
                    file_index,
                    global_start,
                    global_start + ep.length,
                )
                progress.update()
    finally:
        progress.close()

    if offset != sum(plan.episodes[old].length for old in plan.keep):
        raise AssertionError("internal total frame accounting error")
    return _DataRewrite(
        episode_stats=episode_stats,
        values_by_feature=dict(values_by_feature),
        data_refs=data_refs,
    )


def _ffmpeg_command(job: VideoJob, plan: CleanPlan, out: Path, fps: float) -> list[str]:
    ordered_old = sorted(job.start_frames, key=lambda old: job.start_frames[old])
    parts = []
    for old in ordered_old:
        start = job.start_frames[old]
        length = plan.episodes[old].length
        parts.append(f"between(n\\,{start}\\,{start + length - 1})")
    expression = "+".join(parts)
    output = out / job.output_rel
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(job.source),
        "-vf",
        f"select='{expression}',setpts=N/({fps}*TB)",
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        "-an",
        "-c:v",
        plan.vcodec,
    ]
    if plan.vcodec == "libx264":
        cmd.extend(["-preset", "veryfast", "-crf", "18"])
    elif plan.vcodec == "libsvtav1":
        cmd.extend(["-preset", "8", "-crf", "30"])
    cmd.extend(["-pix_fmt", job.pix_fmt, str(output)])
    return cmd


def _check_output_video(path: Path, expected_frames: int, target_codec: str, fps: float) -> None:
    probe = probe_video(path)
    if probe.error is not None:
        raise ValueError(f"output video {path} is unreadable: {probe.error}")
    if probe.nb_frames is None:
        raise ValueError(f"output video {path} has no frame count")
    if probe.nb_frames != expected_frames:
        raise ValueError(
            f"output video {path}: nb_frames={probe.nb_frames}, expected {expected_frames}"
        )
    if (probe.codec or "").lower() != target_codec:
        raise ValueError(
            f"output video {path}: codec={probe.codec}, expected {target_codec}"
        )
    if probe.fps is not None and abs(probe.fps - fps) > 1e-3:
        raise ValueError(f"output video {path}: fps={probe.fps}, expected {fps}")


def _write_videos(plan: CleanPlan, out: Path, reporter: Reporter) -> None:
    fps = float(plan.info["fps"])
    progress = reporter.bar(len(plan.video_jobs), "video files")
    try:
        for job in plan.video_jobs:
            output = out / job.output_rel
            output.parent.mkdir(parents=True, exist_ok=True)
            reporter.info(
                f"  video {job.camera}/{job.output_rel.name}: {job.mode} "
                f"({len(job.kept_old)} ep, {job.expected_frames} frames) - {job.reason}"
            )
            if job.mode == "copy":
                shutil.copy2(job.source, output)
            else:
                cmd = _ffmpeg_command(job, plan, out, fps)
                completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg failed for {job.source} -> {output}: "
                        f"{completed.stderr.strip() or completed.returncode}"
                    )
            _check_output_video(output, job.expected_frames, plan.target_codec, fps)
            progress.update()
    finally:
        progress.close()


def _stat_value_for_arrow(values: np.ndarray, key: str) -> list:
    flat = np.asarray(values).reshape(-1)
    if key == "count":
        return [int(value) for value in flat]
    return [float(value) for value in flat]


def _write_episode_metadata(
    plan: CleanPlan,
    out: Path,
    data: _DataRewrite,
) -> tuple[dict[str, list[dict[str, np.ndarray]]], list[str]]:
    source_table = _load_episode_table(plan.source)
    source_rows = {
        int(row["episode_index"]): row for row in source_table.to_pylist()
    }
    cameras = video_cameras(plan.info)
    image_values: dict[str, list[dict[str, np.ndarray]]] = {camera: [] for camera in cameras}
    notes: list[str] = []
    rows: list[dict] = []

    for old in plan.keep:
        if old not in source_rows:
            raise ValueError(f"ep{old} missing from source episode metadata")
        source_row = source_rows[old]
        row = dict(source_row)
        new_index = plan.old_to_new[old]
        chunk, file, from_index, to_index = data.data_refs[old]
        row["episode_index"] = new_index
        row["data/chunk_index"] = chunk
        row["data/file_index"] = file
        row["dataset_from_index"] = from_index
        row["dataset_to_index"] = to_index
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        for camera in cameras:
            job = plan.video_by_episode_camera[(old, camera)]
            from_s, to_s = job.segments[old]
            row[f"videos/{camera}/chunk_index"] = job.chunk_index
            row[f"videos/{camera}/file_index"] = job.file_index
            row[f"videos/{camera}/from_timestamp"] = float(from_s)
            row[f"videos/{camera}/to_timestamp"] = float(to_s)
            image_stats = image_stats_from_row(source_row, camera)
            if image_stats is not None:
                image_values[camera].append(image_stats)

        for feature, feature_stats in data.episode_stats[old].items():
            for key in STAT_KEYS:
                row[f"stats/{feature}/{key}"] = _stat_value_for_arrow(feature_stats[key], key)
        rows.append(row)

    for camera in cameras:
        if len(image_values[camera]) != len(plan.keep):
            notes.append(
                f"{camera}: source meta/episodes lacks a complete set of image stats; "
                "per-episode values were kept as-is where present, but no global image stats were written"
            )
            image_values[camera] = []

    # Build the output schema from the source schema, appending any low-dim
    # stats columns that the source metadata did not have.
    source_schema = source_table.schema
    fields = list(source_schema)
    field_names = {field.name for field in fields}
    extra_fields: list[pa.Field] = []
    for feature in plan.lowdim_features:
        for key in STAT_KEYS:
            name = f"stats/{feature}/{key}"
            if name in field_names:
                continue
            typ = pa.list_(pa.int64()) if key == "count" else pa.list_(pa.float64())
            extra_fields.append(pa.field(name, typ))
            field_names.add(name)
    for name in ("meta/episodes/chunk_index", "meta/episodes/file_index"):
        if name not in field_names:
            extra_fields.append(pa.field(name, pa.int64()))
            field_names.add(name)
    fields.extend(extra_fields)
    arrays = [
        pa.array([row.get(field.name) for row in rows], type=field.type)
        for field in fields
    ]
    schema = pa.schema(fields, metadata=source_schema.metadata)
    output_table = pa.Table.from_arrays(arrays, schema=schema)
    target = out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, target)
    return image_values, notes


def _write_info_tasks_stats(
    plan: CleanPlan,
    out: Path,
    data: _DataRewrite,
    image_values: dict[str, list[dict[str, np.ndarray]]],
) -> dict[str, str]:
    info = json.loads(json.dumps(plan.info))
    total_frames = sum(plan.episodes[old].length for old in plan.keep)
    info["total_episodes"] = len(plan.keep)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(plan.new_tasks)
    info["splits"] = {"train": f"0:{len(plan.keep)}"}
    for camera in video_cameras(info):
        info["features"][camera].setdefault("info", {})["video.codec"] = plan.target_codec
    (out / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_tasks_path = plan.source / "meta" / "tasks.parquet"
    source_tasks_schema = pq.read_schema(source_tasks_path)
    task_names = {index: name for name, index in plan.new_tasks.items()}
    tasks_table = pa.table(
        {
            "task_index": pa.array(
                list(range(len(task_names))), type=pa.int64()
            ),
            "task": pa.array([task_names[i] for i in range(len(task_names))], type=pa.string()),
        }
    )
    if source_tasks_schema.metadata is not None:
        tasks_table = tasks_table.replace_schema_metadata(source_tasks_schema.metadata)
    pq.write_table(tasks_table, out / "meta" / "tasks.parquet")

    stats_payload: dict[str, dict[str, list]] = {}
    for feature, spec in info.get("features", {}).items():
        if spec.get("dtype") == "video":
            if image_values.get(feature):
                aggregate = aggregate_image_stats(image_values[feature])
                stats_payload[feature] = stats_to_json(aggregate)
            continue
        if feature not in data.values_by_feature:
            continue
        values = np.concatenate(data.values_by_feature[feature], axis=0)
        stats_payload[feature] = stats_to_json(compute_stats(values))
    (out / "meta" / "stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats_payload


def _format_issue(issue) -> str:
    scope = "dataset" if issue.episode_index is None else f"ep{issue.episode_index}"
    return f"{scope} {issue.issue_id}[{issue.severity}]: {issue.detail}"


def _verify_output(plan: CleanPlan, out: Path) -> list[str]:
    issues: list[str] = []
    try:
        info = load_info(out)
        tasks = load_tasks(out)
        episodes = load_episodes(out, info)
    except Exception as exc:  # noqa: BLE001
        return [f"cannot reload output metadata: {exc}"]
    for issue in check_metadata_totals(out, info, tasks, episodes):
        issues.append(_format_issue(issue))
    for issue in check_inventory(out, info, episodes):
        issues.append(_format_issue(issue))
    for ep in episodes:
        try:
            table = read_data_table(resolve_data_path(out, info, ep))
            arrays = extract_episode_arrays(table, ep)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"ep{ep.episode_index}: cannot read output data: {exc}")
            continue
        for issue in check_episode_structure(ep, arrays, tasks):
            issues.append(_format_issue(issue))

    total_frames = sum(ep.length for ep in episodes)
    try:
        stats_payload = json.loads((out / "meta" / "stats.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        issues.append(f"cannot read meta/stats.json: {exc}")
        stats_payload = {}
    for feature in plan.lowdim_features:
        if feature not in stats_payload:
            issues.append(f"meta/stats.json missing feature {feature!r}")
            continue
        count = stats_payload[feature].get("count", [None])[0]
        if count != total_frames:
            issues.append(
                f"meta/stats.json {feature}.count={count}, expected {total_frames}"
            )
    if stats_payload:
        for feature, expected_len in (("action", 20), ("observation.state", 48)):
            if feature in stats_payload and len(stats_payload[feature].get("min", [])) != expected_len:
                issues.append(
                    f"meta/stats.json {feature} has {len(stats_payload[feature].get('min', []))} dims, "
                    f"expected {expected_len}"
                )
        if "index" in stats_payload:
            if stats_payload["index"]["min"][0] != 0:
                issues.append("meta/stats.json index.min must be 0")
            if stats_payload["index"]["max"][0] != total_frames - 1:
                issues.append("meta/stats.json index.max must be total_frames-1")
        if "episode_index" in stats_payload:
            if stats_payload["episode_index"]["min"][0] != 0:
                issues.append("meta/stats.json episode_index.min must be 0")
            if stats_payload["episode_index"]["max"][0] != len(episodes) - 1:
                issues.append("meta/stats.json episode_index.max must be N-1")

    # Per-episode re-indexed stats must not be stale copies.
    ep_meta = pq.read_table(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    for row in ep_meta.to_pylist():
        ep_index = int(row["episode_index"])
        from_index = int(row["dataset_from_index"])
        to_index = int(row["dataset_to_index"])
        checks = {
            "stats/episode_index/min": ep_index,
            "stats/episode_index/max": ep_index,
            "stats/index/min": from_index,
            "stats/index/max": to_index - 1,
            "stats/index/count": to_index - from_index,
        }
        for column, expected in checks.items():
            if column not in row:
                continue
            actual = int(row[column][0])
            if actual != expected:
                issues.append(
                    f"ep{ep_index} {column}={actual}, expected {expected}"
                )
    # Global image counts must equal the sum of retained per-episode counts.
    for camera in video_cameras(info):
        count_key = f"stats/{camera}/count"
        if camera not in stats_payload or count_key not in ep_meta.column_names:
            continue
        expected = sum(int(value[0]) for value in ep_meta.column(count_key).to_pylist())
        actual = int(stats_payload[camera]["count"][0])
        if actual != expected:
            issues.append(
                f"meta/stats.json {camera}.count={actual}, expected {expected}"
            )

    video_probes: dict[Path, VideoProbe] = {}
    for job in plan.video_jobs:
        output = out / job.output_rel
        try:
            _check_output_video(output, job.expected_frames, plan.target_codec, float(info["fps"]))
            video_probes[output] = probe_video(output)
        except Exception as exc:  # noqa: BLE001
            issues.append(str(exc))

    fps = float(info["fps"])
    by_new = {ep.episode_index: ep for ep in episodes}
    for job in plan.video_jobs:
        output = out / job.output_rel
        probe = video_probes.get(output)
        for old in job.kept_old:
            new = plan.old_to_new[old]
            ep = by_new.get(new)
            if ep is None:
                issues.append(f"video plan references missing output ep{new}")
                continue
            for camera in (job.camera,):
                seg = ep.videos.get(camera)
                if seg is None:
                    issues.append(f"ep{new} has no {camera} video metadata")
                    continue
                expected_from, expected_to = job.segments[old]
                if abs(seg.from_timestamp - expected_from) > 1e-6 or abs(
                    seg.to_timestamp - expected_to
                ) > 1e-6:
                    issues.append(
                        f"ep{new} {camera} segment [{seg.from_timestamp}, {seg.to_timestamp}] "
                        f"!= planned [{expected_from}, {expected_to}]"
                    )
                if abs((seg.to_timestamp - seg.from_timestamp) * fps - ep.length) > 0.5:
                    issues.append(
                        f"ep{new} {camera} segment duration != length/fps"
                    )
                if (
                    probe is not None
                    and probe.nb_frames is not None
                    and seg.to_timestamp > probe.nb_frames / fps + 0.5 / fps
                ):
                    issues.append(
                        f"ep{new} {camera} segment end {seg.to_timestamp:.6f}s "
                        f"exceeds output video duration"
                    )
    return issues


def _write_deleted_csv(plan: CleanPlan, out: Path) -> Path:
    path = out / "deleted_episodes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "episode_index",
                "length",
                "data_file",
                "video_files",
                "reasons",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in plan.deleted_rows:
            writer.writerow(
                {
                    "episode_index": row.episode_index,
                    "length": row.length,
                    "data_file": row.data_file,
                    "video_files": row.video_files,
                    "reasons": row.reasons,
                }
            )
    return path


def _mapping_video_text(plan: CleanPlan, old: int) -> str:
    parts = []
    for camera in video_cameras(plan.info):
        job = plan.video_by_episode_camera[(old, camera)]
        from_s, to_s = job.segments[old]
        parts.append(f"{camera}={job.output_rel}:{from_s:.6f}-{to_s:.6f}")
    return "; ".join(parts)


def _write_mapping_csv(plan: CleanPlan, out: Path) -> Path:
    path = out / "index_mapping.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "old_episode_index",
                "new_episode_index",
                "length",
                "data_file",
                "video_files",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for old in plan.keep:
            new = plan.old_to_new[old]
            chunk = new // int(plan.info.get("chunks_size", 1000))
            file = new % int(plan.info.get("chunks_size", 1000))
            data_rel = plan.info["data_path"].format(chunk_index=chunk, file_index=file)
            writer.writerow(
                {
                    "old_episode_index": old,
                    "new_episode_index": new,
                    "length": plan.episodes[old].length,
                    "data_file": data_rel,
                    "video_files": _mapping_video_text(plan, old),
                }
            )
    return path


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_report(
    plan: CleanPlan,
    out: Path,
    *,
    stats_payload: dict,
    image_notes: list[str],
    verification_issues: list[str],
) -> Path:
    total_frames = sum(plan.episodes[old].length for old in plan.keep)
    lines: list[str] = []
    lines.append("# 数据清洗报告")
    lines.append("")
    lines.append(f"- 生成时间：`{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    lines.append(f"- 源数据集：`{plan.source}`（只读，未修改）")
    lines.append(f"- 输出数据集：`{out}`")
    lines.append(
        f"- episode：删除 **{len(plan.deleted)}**，保留 **{len(plan.keep)}**；"
        f"帧数：**{total_frames}**"
    )
    lines.append(f"- 视频编码：`{plan.vcodec}`（`{plan.target_codec}`）")
    lines.append("")

    lines.append("## 删除列表")
    lines.append("")
    lines.append("| old episode | length | 原因 |")
    lines.append("|---|---:|---|")
    for row in plan.deleted_rows:
        reason = row.reasons
        if len(reason) > 300:
            reason = reason[:297] + "..."
        lines.append(f"| {row.episode_index} | {row.length} | {_md(reason)} |")
    lines.append("")
    lines.append(
        "完整删除信息见 `deleted_episodes.csv`；其中 `data_file` / `video_files` "
        "是删除前的源数据集引用。"
    )
    lines.append("")

    lines.append("## 重编号")
    lines.append("")
    lines.append("连续区间的 old -> new 映射：")
    lines.append("")
    lines.append("| old episodes | new episodes |")
    lines.append("|---|---|")
    for old_start, old_end, new_start, new_end in _mapping_runs(plan.keep, plan.old_to_new):
        old_text = f"{old_start}" if old_start == old_end else f"{old_start}..{old_end}"
        new_text = f"{new_start}" if new_start == new_end else f"{new_start}..{new_end}"
        lines.append(f"| {old_text} | {new_text} |")
    lines.append("")
    lines.append(
        "逐集映射见 `index_mapping.csv`；其中 `data_file` / `video_files` "
        "是输出数据集路径和时间段。"
    )
    lines.append("")

    lines.append("## 数据文件")
    lines.append("")
    lines.append(
        f"- 每个保留 episode 写一个 data parquet，共 **{len(plan.keep)}** 个；"
        "`data/chunk-000/file-000..N-1.parquet` 连续编号。"
    )
    lines.append("- `frame_index`、`timestamp`、action/state 原值不变。")
    lines.append("- `episode_index`、全局 `index`、`task_index` 已按保留集重映射。")
    lines.append("")

    lines.append("## 视频文件")
    lines.append("")
    lines.append("| camera | file | 方式 | 保留 episode 数 | 输出帧数 | 说明 |")
    lines.append("|---|---|---|---:|---:|---|")
    for job in plan.video_jobs:
        lines.append(
            f"| `{_md(job.camera)}` | `{_md(job.output_rel.name)}` | "
            f"{'复制' if job.mode == 'copy' else '重编码'} | {len(job.kept_old)} | "
            f"{job.expected_frames} | {_md(job.reason)} |"
        )
    if plan.skipped_video_files:
        lines.append("")
        lines.append("未写出的源视频文件：")
        for text in plan.skipped_video_files:
            lines.append(f"- {_md(text)}")
    lines.append("")
    lines.append(
        "每个 episode 的 `videos/<camera>/from_timestamp`、`to_timestamp` 已按输出文件重新计算；"
        "复制分支保持源时间戳，重编码分支从 0 开始连续累加。"
    )
    lines.append("")

    lines.append("## 统计量")
    lines.append("")
    lines.append(
        "- 低维特征（action、observation.state、timestamp、frame_index、"
        "episode_index、index、task_index）**逐集精确重算**，"
        "全局 stats 也由所有保留行精确重算。"
    )
    lines.append(
        "- 图像特征：逐集 stats 复用源 `meta/episodes` 的值；"
        "全局 stats 按像素数加权聚合（min/max、加权 mean/quantile、合并方差）。"
    )
    lines.append(
        "- 图像 stats 不是逐像素重算：视频重编码只改变压缩，不改变帧内容。"
    )
    if image_notes:
        for note in image_notes:
            lines.append(f"- 注意：{_md(note)}")
    lines.append(
        f"- `meta/stats.json` 中共写入 **{len(stats_payload)}** 个特征。"
    )
    lines.append("")

    lines.append("## 校验")
    lines.append("")
    if verification_issues:
        lines.append(f"**失败**：发现 {len(verification_issues)} 个问题。")
        lines.append("")
        for text in verification_issues[:50]:
            lines.append(f"- {_md(text)}")
        if len(verification_issues) > 50:
            lines.append(f"- ... 另有 {len(verification_issues) - 50} 条")
    else:
        lines.append(
            "通过：输出数据集的 episode/index 连续性、data 行数、任务映射、"
            "视频帧数/编码、低维统计量和图像统计 count 均通过检查。"
        )
    lines.append("")

    if plan.warnings:
        lines.append("## 警告")
        lines.append("")
        for warning in plan.warnings:
            lines.append(f"- {_md(warning)}")
        lines.append("")

    lines.append("## 已知限制")
    lines.append("")
    lines.append(
        "- 源数据集只读；所有改动只写入输出副本。"
    )
    lines.append(
        "- 数据文件被压缩为每集一个文件，与 LeRobot 原实现“保留源 file_index”不同；"
        "这样删除后 data 文件编号连续。"
    )
    lines.append(
        "- 视频保留源文件 chunk/file 编号；若某源视频文件中的 episode 全部被删除，"
        "输出会留下编号空洞（当前数据集不会发生）。"
    )
    lines.append(
        "- 图像 stats 复用源逐集值；若需要逐像素精确统计，需要额外解码全部视频。"
    )
    lines.append(
        "- 仅支持 CFR 且 episode 边界可按 `round(timestamp * fps)` 映射到源帧的视频。"
    )
    lines.append("")

    path = out / "cleaning_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_clean(
    plan: CleanPlan,
    out: Path,
    *,
    overwrite: bool = False,
    reporter: Reporter | None = None,
    prepare: bool = True,
) -> CleanResult:
    reporter = reporter or Reporter(total_stages=6)
    cameras = video_cameras(plan.info)
    if prepare:
        prepare_clean_output(out, overwrite, cameras)

    reporter.stage("重写 data parquet（过滤、重编号、逐集统计）")
    data = _write_data(plan, out, reporter)

    reporter.stage("重写视频（复制或按帧区间重编码）")
    _write_videos(plan, out, reporter)

    reporter.stage("重写 meta/episodes")
    image_values, image_notes = _write_episode_metadata(plan, out, data)

    reporter.stage("写 info.json / tasks.parquet / stats.json")
    stats_payload = _write_info_tasks_stats(plan, out, data, image_values)

    reporter.stage("校验输出数据集")
    verification_issues = _verify_output(plan, out)

    reporter.stage("写清洗报告和映射表")
    deleted_csv = _write_deleted_csv(plan, out)
    mapping_csv = _write_mapping_csv(plan, out)
    report_path = _write_report(
        plan,
        out,
        stats_payload=stats_payload,
        image_notes=image_notes,
        verification_issues=verification_issues,
    )

    return CleanResult(
        out=out,
        deleted_count=len(plan.deleted),
        kept_count=len(plan.keep),
        total_frames=sum(plan.episodes[old].length for old in plan.keep),
        report_path=report_path,
        deleted_csv=deleted_csv,
        mapping_csv=mapping_csv,
        verification_issues=verification_issues,
        warnings=plan.warnings,
    )


def format_dry_run(plan: CleanPlan) -> str:
    total_frames = sum(plan.episodes[old].length for old in plan.keep)
    lines: list[str] = []
    lines.append("dclean clean (dry-run): no files will be written")
    lines.append(f"source: {plan.source}")
    lines.append(
        f"delete {len(plan.deleted)} / {len(plan.episodes)} episodes; "
        f"keep {len(plan.keep)} episodes, {total_frames} frames"
    )
    lines.append(
        "deleted: " + ", ".join(str(index) for index in plan.deleted)
    )
    for row in plan.deleted_rows:
        reason = row.reasons if len(row.reasons) <= 180 else row.reasons[:177] + "..."
        lines.append(f"  ep{row.episode_index}: {reason}")
    lines.append("old -> new ranges:")
    for old_start, old_end, new_start, new_end in _mapping_runs(plan.keep, plan.old_to_new):
        old_text = f"{old_start}" if old_start == old_end else f"{old_start}..{old_end}"
        new_text = f"{new_start}" if new_start == new_end else f"{new_start}..{new_end}"
        lines.append(f"  {old_text} -> {new_text}")
    lines.append(f"data files: {len(plan.keep)} (one per retained episode)")
    lines.append(f"video codec: {plan.vcodec} ({plan.target_codec})")
    lines.append("video plan:")
    for job in plan.video_jobs:
        lines.append(
            f"  {job.camera}/{job.output_rel.name}: {job.mode}, "
            f"{len(job.kept_old)} ep, {job.expected_frames} frames - {job.reason}"
        )
    if plan.skipped_video_files:
        lines.append("video files skipped (all episodes deleted):")
        for text in plan.skipped_video_files:
            lines.append(f"  {text}")
    if plan.warnings:
        lines.append("warnings:")
        for warning in plan.warnings:
            lines.append(f"  {warning}")
    return "\n".join(lines) + "\n"
