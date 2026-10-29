"""Read-only action/video timestamp alignment check.

LeRobot v3.0 stores per-episode/camera ``from_timestamp``/``to_timestamp`` in
the episode metadata, while per-frame video timestamps live in the MP4 as PTS.
This module extracts packet PTS once per video file, maps each episode segment
to its PTS sequence, and compares it with the action ``timestamp`` column.

It does not modify data and does not resample. Packet PTS is used instead of
decoded-frame PTS because demuxing is ~100x cheaper and, for CFR MP4, one
packet corresponds to one frame.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .dataset import EpisodeArrays, EpisodeMeta, resolve_video_path, video_cameras
from .progress import ProgressBar
from .scoring import Issue, dataset_issue, make_issue
from .video import probe_video


@dataclass
class AVSyncRow:
    episode_index: int
    camera: str
    length: int
    video_frames: int
    action_fps: float | None
    video_fps: float | None
    action_start_s: float | None
    video_first_pts: float | None
    video_last_pts: float | None
    start_offset_s: float | None
    max_drift_s: float | None
    max_drift_frame: int | None
    pts_gap_max_dev_s: float | None
    status: str
    issue: str


def extract_packet_pts(path: Path) -> tuple[np.ndarray, str | None]:
    """Return sorted video packet PTS values (seconds) and an optional error."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return np.zeros(0, dtype=np.float64), completed.stderr.strip() or "ffprobe failed"
    values: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        if not token or token == "N/A":
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        return np.zeros(0, dtype=np.float64), "no packet pts_time values"
    pts = np.unique(np.asarray(values, dtype=np.float64))
    return pts, None


def compare_segment(
    ep: EpisodeMeta,
    camera: str,
    pts: np.ndarray,
    action_ts: np.ndarray,
    from_s: float,
    to_s: float,
    video_fps: float,
    cfg: Config,
) -> tuple[Issue | None, AVSyncRow]:
    """Compare one episode/camera PTS segment with the action timestamps."""
    eps = 0.25 / video_fps if video_fps > 0 else 1e-3
    segment = pts[(pts >= from_s - eps) & (pts < to_s - eps)]
    n = int(segment.size)
    action_dt = np.diff(action_ts) if action_ts.size >= 2 else np.zeros(0)
    action_fps = (
        1.0 / float(np.median(action_dt))
        if action_dt.size and float(np.median(action_dt)) > 0
        else float(cfg.fps)
    )
    action_start = float(action_ts[0]) if action_ts.size else None
    tol_s = cfg.av_sync_tol_frames / action_fps
    warn_s = cfg.av_sync_warn_frames / action_fps
    gap_tol_s = cfg.av_sync_pts_gap_tol_frames / video_fps if video_fps > 0 else 0.0

    start_offset = float(segment[0] - from_s) if n else None
    max_drift = None
    max_drift_frame = None
    pts_gap_max_dev = None
    if n >= 2:
        pts_gap_max_dev = float(np.max(np.abs(np.diff(segment) - 1.0 / video_fps)))
    if n == ep.length and action_ts.size == ep.length:
        video_rel = segment - segment[0]
        action_rel = action_ts - action_ts[0]
        drift = video_rel - action_rel
        max_drift_frame = int(np.argmax(np.abs(drift)))
        max_drift = float(abs(drift[max_drift_frame]))

    status = "ok"
    reasons: list[str] = []
    if n != ep.length:
        status = "fail"
        reasons.append(f"video_frames={n} != length={ep.length}")
    if action_start is not None and abs(action_start) > tol_s:
        status = "fail"
        reasons.append(f"action_start={action_start:.4f}s")
    elif action_start is not None and abs(action_start) > warn_s:
        status = "warn" if status == "ok" else status
        reasons.append(f"action_start={action_start:.4f}s")
    if start_offset is not None and abs(start_offset) > tol_s:
        status = "fail"
        reasons.append(f"video_start_offset={start_offset * 1000:.1f}ms")
    elif start_offset is not None and abs(start_offset) > warn_s:
        status = "warn" if status == "ok" else status
        reasons.append(f"video_start_offset={start_offset * 1000:.1f}ms")
    if max_drift is not None and max_drift > tol_s:
        status = "fail"
        reasons.append(f"max_drift={max_drift * 1000:.1f}ms at frame {max_drift_frame}")
    elif max_drift is not None and max_drift > warn_s:
        status = "warn" if status == "ok" else status
        reasons.append(f"max_drift={max_drift * 1000:.1f}ms at frame {max_drift_frame}")
    if pts_gap_max_dev is not None and pts_gap_max_dev > gap_tol_s:
        status = "warn" if status == "ok" else status
        reasons.append(f"pts_gap_dev={pts_gap_max_dev * 1000:.1f}ms")

    issue = None
    if status != "ok":
        detail = (
            f"{camera}: action_fps={action_fps:.3f}, video_fps={video_fps:.3f}, "
            f"length={ep.length}, video_frames={n}; " + "; ".join(reasons)
        )
        issue = make_issue(
            "l1.av_timestamp_mismatch",
            status,
            ep.episode_index,
            0,
            max(0, ep.length - 1),
            1,
            detail,
        )

    row = AVSyncRow(
        episode_index=ep.episode_index,
        camera=camera,
        length=ep.length,
        video_frames=n,
        action_fps=round(action_fps, 6),
        video_fps=round(video_fps, 6),
        action_start_s=action_start,
        video_first_pts=float(segment[0]) if n else None,
        video_last_pts=float(segment[-1]) if n else None,
        start_offset_s=start_offset,
        max_drift_s=max_drift,
        max_drift_frame=max_drift_frame,
        pts_gap_max_dev_s=pts_gap_max_dev,
        status=status,
        issue="l1.av_timestamp_mismatch" if issue else "",
    )
    return issue, row


def check_av_sync(
    root: Path,
    info: dict,
    episodes: list[EpisodeMeta],
    arrays_by_episode: dict[int, EpisodeArrays],
    cfg: Config,
    *,
    progress: ProgressBar | None = None,
) -> tuple[list[Issue], list[Issue], list[AVSyncRow]]:
    """Return (dataset issues, episode issues, rows)."""
    dataset_issues: list[Issue] = []
    episode_issues: list[Issue] = []
    rows: list[AVSyncRow] = []

    if not cfg.av_sync_enabled:
        return dataset_issues, episode_issues, rows

    usages: dict[Path, list[tuple[EpisodeMeta, str, float, float]]] = {}
    for ep in episodes:
        for camera in video_cameras(info):
            seg = ep.videos.get(camera)
            if seg is None:
                continue
            path = resolve_video_path(root, info, ep, camera)
            if path.is_file() and path.stat().st_size > 0:
                usages.setdefault(path, []).append(
                    (ep, camera, float(seg.from_timestamp), float(seg.to_timestamp))
                )

    if not usages:
        return dataset_issues, episode_issues, rows

    for path in sorted(usages, key=str):
        pts, error = extract_packet_pts(path)
        if error is not None or pts.size < 2:
            dataset_issues.append(
                dataset_issue(
                    "l1.av_pts_unreadable",
                    "warn",
                    f"{path.name}: cannot read packet PTS ({error or 'too few PTS'})",
                )
            )
            if progress is not None:
                progress.update()
            continue
        probe = probe_video(path)
        if probe.nb_frames is not None and abs(pts.size - probe.nb_frames) > 1:
            dataset_issues.append(
                dataset_issue(
                    "l1.av_pts_count",
                    "warn",
                    f"{path.name}: packet PTS count {pts.size} != nb_frames {probe.nb_frames}",
                )
            )
        pts_dt = np.diff(pts)
        video_fps = (
            1.0 / float(np.median(pts_dt))
            if pts_dt.size and float(np.median(pts_dt)) > 0
            else float(info.get("fps", cfg.fps))
        )
        for ep, camera, from_s, to_s in usages[path]:
            arrays = arrays_by_episode.get(ep.episode_index)
            if arrays is None or arrays.num_rows == 0:
                continue
            issue, row = compare_segment(
                ep, camera, pts, arrays.timestamp, from_s, to_s, video_fps, cfg
            )
            rows.append(row)
            if issue is not None:
                episode_issues.append(issue)
        if progress is not None:
            progress.update()

    flagged = [row for row in rows if row.status != "ok"]
    dataset_issues.append(
        dataset_issue(
            "l1.av_sync_baseline",
            "info",
            f"compared packet PTS with action timestamps for {len(usages)} video file(s), "
            f"{len(rows)} episode/camera row(s); {len(flagged)} mismatch(es)",
        )
    )
    return dataset_issues, episode_issues, rows
