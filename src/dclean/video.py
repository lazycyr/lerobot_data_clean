"""ffprobe-based video metadata checks (no frame decoding)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .dataset import EpisodeArrays, EpisodeMeta, resolve_video_path, video_cameras
from .progress import ProgressBar
from .scoring import Issue, dataset_issue, make_issue


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    nb_frames: int | None
    duration: float | None
    pix_fmt: str | None
    estimated_frames: bool
    error: str | None = None


def parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_f = float(denominator)
            if denominator_f == 0:
                return None
            return float(numerator) / denominator_f
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_video(path: Path, timeout_s: float = 20.0) -> VideoProbe:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VideoProbe(path, None, None, None, None, None, None, None, False, str(exc))
    if completed.returncode != 0:
        error = completed.stderr.strip() or f"ffprobe exited {completed.returncode}"
        return VideoProbe(path, None, None, None, None, None, None, None, False, error)
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return VideoProbe(path, None, None, None, None, None, None, None, False, f"bad ffprobe JSON: {exc}")

    streams = payload.get("streams") or []
    if not streams:
        return VideoProbe(path, None, None, None, None, None, None, None, False, "no video stream")
    stream = streams[0]
    fmt = payload.get("format") or {}

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    duration = _as_float(stream.get("duration")) or _as_float(fmt.get("duration"))
    nb_frames = _as_int(stream.get("nb_frames"))
    estimated = False
    if nb_frames is None and duration is not None and fps is not None:
        nb_frames = int(round(duration * fps))
        estimated = True

    return VideoProbe(
        path=path,
        codec=(stream.get("codec_name") or None),
        width=_as_int(stream.get("width")),
        height=_as_int(stream.get("height")),
        fps=fps,
        nb_frames=nb_frames,
        duration=duration,
        pix_fmt=(stream.get("pix_fmt") or None),
        estimated_frames=estimated,
    )


def _declared(info: dict, camera: str) -> dict:
    feature = info.get("features", {}).get(camera, {})
    meta = feature.get("info", {}) or {}
    shape = feature.get("shape") or [None, None, None]
    return {
        "codec": str(meta.get("video.codec", "")).lower(),
        "width": meta.get("video.width", shape[1] if len(shape) > 1 else None),
        "height": meta.get("video.height", shape[0] if len(shape) > 0 else None),
        "fps": meta.get("video.fps", info.get("fps")),
        "pix_fmt": meta.get("video.pix_fmt"),
    }


def _episode_range(episodes: list[EpisodeMeta]) -> str:
    if not episodes:
        return "no episodes"
    indices = sorted(ep.episode_index for ep in episodes)
    return f"ep{indices[0]}" if len(indices) == 1 else f"ep{indices[0]}-{indices[-1]}"


def check_video(
    root: Path,
    info: dict,
    episodes: list[EpisodeMeta],
    *,
    all_episodes: list[EpisodeMeta] | None = None,
    arrays_by_episode: dict[int, EpisodeArrays],
    cfg: Config,
    progress: ProgressBar | None = None,
) -> tuple[list[Issue], list[Issue]]:
    """Return (dataset_issues, episode_issues).

    ``episodes`` scopes per-episode checks; ``all_episodes`` is used for
    file-level frame counts and segment continuity so subset checks do not
    produce false gaps.
    """
    dataset_issues: list[Issue] = []
    episode_issues: list[Issue] = []
    all_episodes = all_episodes if all_episodes is not None else episodes
    selected_set = {ep.episode_index for ep in episodes}

    if shutil.which("ffprobe") is None:
        dataset_issues.append(
            dataset_issue(
                "l3.ffprobe_missing",
                "warn",
                "ffprobe not found; video metadata checks were skipped",
            )
        )
        return dataset_issues, episode_issues

    all_usages: dict[Path, list[tuple[EpisodeMeta, str]]] = {}
    for ep in all_episodes:
        for camera in video_cameras(info):
            if camera not in ep.videos:
                continue
            path = resolve_video_path(root, info, ep, camera)
            if path.is_file() and path.stat().st_size > 0:
                all_usages.setdefault(path, []).append((ep, camera))

    usages: dict[Path, list[tuple[EpisodeMeta, str]]] = {}
    for ep in episodes:
        for camera in video_cameras(info):
            if camera not in ep.videos:
                continue
            path = resolve_video_path(root, info, ep, camera)
            if path.is_file() and path.stat().st_size > 0:
                usages.setdefault(path, []).append((ep, camera))

    probes: dict[Path, VideoProbe] = {}
    for path in sorted(usages, key=str):
        probes[path] = probe_video(path)
        if progress is not None:
            progress.update()

    for path, used in sorted(usages.items(), key=lambda item: str(item[0])):
        probe = probes[path]
        camera = used[0][1]
        declared = _declared(info, camera)
        all_used = all_usages.get(path, used)
        affected = sorted({ep.episode_index for ep, _ in all_used})
        affected_text = f"ep{affected[0]}" if len(affected) == 1 else f"ep{affected[0]}-{affected[-1]}"
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

        if probe.error is not None:
            dataset_issues.append(
                dataset_issue("l3.video_probe", "fail", f"{rel}: ffprobe failed: {probe.error}")
            )
            continue

        if declared["codec"] and probe.codec and declared["codec"] != probe.codec.lower():
            severity = cfg.codec_mismatch_status
            dataset_issues.append(
                dataset_issue(
                    "l3.video_codec_mismatch",
                    severity,
                    f"{rel}: actual codec={probe.codec}, declared={declared['codec']} "
                    f"(affects {affected_text})",
                )
            )

        if (
            declared["width"] is not None
            and probe.width is not None
            and int(declared["width"]) != int(probe.width)
        ) or (
            declared["height"] is not None
            and probe.height is not None
            and int(declared["height"]) != int(probe.height)
        ):
            dataset_issues.append(
                dataset_issue(
                    "l3.video_resolution_mismatch",
                    "warn",
                    f"{rel}: actual {probe.width}x{probe.height}, declared "
                    f"{declared['width']}x{declared['height']} (affects {affected_text})",
                )
            )

        if declared["fps"] is not None and probe.fps is not None:
            declared_fps = float(declared["fps"])
            if abs(probe.fps - declared_fps) > 1e-3:
                dataset_issues.append(
                    dataset_issue(
                        "l1.video_fps_mismatch",
                        "warn",
                        f"{rel}: actual fps={probe.fps:.4f}, info.fps={declared_fps:.4f}; "
                        "video and low-dimensional data must share one frame grid",
                    )
                )

        if probe.nb_frames is not None and probe.duration is not None and probe.fps is not None:
            expected_by_duration = probe.duration * probe.fps
            if abs(probe.nb_frames - expected_by_duration) > cfg.frame_count_tol:
                dataset_issues.append(
                    dataset_issue(
                        "l3.video_frame_count",
                        "warn",
                        f"{rel}: nb_frames={probe.nb_frames}, duration*fps={expected_by_duration:.1f}",
                    )
                )

        expected_frames = sum(ep.length for ep, _ in all_used)
        if probe.nb_frames is not None:
            diff = abs(probe.nb_frames - expected_frames)
            if diff > cfg.frame_count_tol:
                severity = "warn" if probe.estimated_frames else "fail"
                dataset_issues.append(
                    dataset_issue(
                        "l1.video_frame_count",
                        severity,
                        f"{rel}: nb_frames={probe.nb_frames}"
                        f"{' (estimated)' if probe.estimated_frames else ''}, "
                        f"sum(length)={expected_frames} (affects {affected_text})",
                    )
                )

    # Per-episode segment checks.
    expected_segment_s = None
    segment_tol_s = cfg.frame_count_tol / cfg.fps
    bad_segments: set[tuple[int, str]] = set()
    for ep in episodes:
        failed_segments: list[str] = []
        for camera in video_cameras(info):
            seg = ep.videos.get(camera)
            if seg is None:
                continue
            path = resolve_video_path(root, info, ep, camera)
            probe = probes.get(path)
            if probe is None or probe.error is not None or probe.duration is None:
                continue
            duration = probe.duration
            bound_tol = 0.5 / cfg.fps
            expected_segment_s = ep.length / cfg.fps
            if (
                seg.from_timestamp < -bound_tol
                or seg.to_timestamp <= seg.from_timestamp
                or seg.to_timestamp > duration + bound_tol
            ):
                failed_segments.append(
                    f"{camera}: [{seg.from_timestamp:.3f}, {seg.to_timestamp:.3f}] "
                    f"outside video duration {duration:.3f}s"
                )
                bad_segments.add((ep.episode_index, camera))
                continue
            if abs((seg.to_timestamp - seg.from_timestamp) - expected_segment_s) > segment_tol_s:
                bad_segments.add((ep.episode_index, camera))
                episode_issues.append(
                    make_issue(
                        "l1.video_segment_duration",
                        "fail",
                        ep.episode_index,
                        0,
                        max(0, ep.length - 1),
                        1,
                        f"{camera}: segment duration "
                        f"{seg.to_timestamp - seg.from_timestamp:.4f}s, expected "
                        f"{expected_segment_s:.4f}s (length={ep.length}, fps={cfg.fps})",
                    )
                )
            arrays = arrays_by_episode.get(ep.episode_index)
            if arrays is not None and arrays.num_rows:
                max_ts = float(np.nanmax(arrays.timestamp))
                if max_ts > seg.to_timestamp + bound_tol:
                    episode_issues.append(
                        make_issue(
                            "l1.video_data_time_mismatch",
                            "warn",
                            ep.episode_index,
                            0,
                            max(0, ep.length - 1),
                            1,
                            f"{camera}: max data timestamp {max_ts:.4f}s exceeds segment end "
                            f"{seg.to_timestamp:.4f}s",
                        )
                    )
        if failed_segments:
            episode_issues.append(
                make_issue(
                    "l1.video_bounds",
                    "fail",
                    ep.episode_index,
                    0,
                    max(0, ep.length - 1),
                    len(failed_segments),
                    "; ".join(failed_segments[:4]),
                )
            )

    # Within each (camera, file), adjacent episode segments should touch.
    # Use all episodes so a subset check does not look like a gap, but only
    # report the right-hand episode when it is in the selected scope.
    groups: dict[tuple[str, int, int], list[EpisodeMeta]] = {}
    for ep in all_episodes:
        for camera, seg in ep.videos.items():
            path = resolve_video_path(root, info, ep, camera)
            if path in probes:
                groups.setdefault((camera, seg.chunk_index, seg.file_index), []).append(ep)
    for (camera, _chunk, _file), group in groups.items():
        group = sorted(group, key=lambda e: e.videos[camera].from_timestamp)
        for left, right in zip(group, group[1:]):
            if right.episode_index not in selected_set:
                continue
            # A bad left segment already explains the discontinuity; do not
            # cascade a second warning onto the next episode.
            if (left.episode_index, camera) in bad_segments:
                continue
            left_seg = left.videos[camera]
            right_seg = right.videos[camera]
            if abs(left_seg.to_timestamp - right_seg.from_timestamp) > 1e-6:
                episode_issues.append(
                    make_issue(
                        "l1.video_segments_not_contiguous",
                        "warn",
                        right.episode_index,
                        0,
                        max(0, right.length - 1),
                        1,
                        f"{camera}: previous {left_seg.to_timestamp:.4f}s != "
                        f"next {right_seg.from_timestamp:.4f}s (ep{left.episode_index}->ep{right.episode_index})",
                    )
                )

    return dataset_issues, episode_issues
