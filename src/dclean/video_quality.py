"""Sampled frame-quality checks for LeRobot v3.0 videos.

v1 is deliberately dependency-light: ffmpeg decodes each unique video file to
small grayscale frames (default 160x120 at 5 fps), and numpy computes:

- mean/std luma           -> black, dark, overexposed frames
- 4-neighbour Laplacian  -> sharpness / blur
- frame-to-frame MAD     -> frozen / duplicate frames

Detected events are mapped back to episode frame ranges through the
``videos/<camera>/from_timestamp`` / ``to_timestamp`` metadata.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .dataset import EpisodeMeta, resolve_video_path, video_cameras
from .progress import ProgressBar
from .scoring import Issue, dataset_issue, make_issue
from .video import probe_video


@dataclass
class SampleSeries:
    times: np.ndarray
    mean_luma: np.ndarray
    std_luma: np.ndarray
    blur: np.ndarray
    frame_diff: np.ndarray
    dark_fraction: np.ndarray
    bright_fraction: np.ndarray
    fps: float
    decode_error: str | None = None


@dataclass
class VideoQualityRow:
    episode_index: int
    camera: str
    samples: int
    black_ratio: float
    dark_ratio: float
    bright_ratio: float
    freeze_ratio: float
    blur_low_ratio: float
    blur_min: float | None
    blur_median: float | None
    luma_min: float | None
    luma_median: float | None
    luma_max: float | None
    status: str
    issues: str


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    return runs


def _to_frame(time_s: float, from_s: float, to_s: float, length: int) -> int:
    if to_s <= from_s or length <= 1:
        return 0
    frame = int(round((time_s - from_s) / (to_s - from_s) * (length - 1)))
    return max(0, min(length - 1, frame))


def extract_samples(path: Path, cfg: Config, native_fps: float) -> SampleSeries:
    """Decode a video file to grayscale frames and compute metrics.

    ``full_frame=True`` (default) processes every decoded frame. Set
    ``sample_fps`` and ``full_frame=False`` for a cheaper sampled pass.
    """
    width = cfg.video_quality_width
    height = cfg.video_quality_height
    fps = native_fps if cfg.video_quality_full_frame else cfg.video_quality_sample_fps
    if fps is None or fps <= 0:
        fps = cfg.fps
    frame_bytes = width * height
    filter_chain = f"scale={width}:{height},format=gray"
    if not cfg.video_quality_full_frame:
        filter_chain = f"fps={fps},{filter_chain}"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        filter_chain,
        "-an",
        "-sn",
        "-dn",
        "-f",
        "rawvideo",
        "-",
    ]
    times: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    blurs: list[float] = []
    diffs: list[float] = []
    darks: list[float] = []
    brights: list[float] = []

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 4,
    )
    previous: np.ndarray | None = None
    index = 0
    assert process.stdout is not None
    while True:
        buffer = process.stdout.read(frame_bytes)
        if not buffer or len(buffer) < frame_bytes:
            break
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width).astype(np.float32)
        laplacian = (
            -4.0 * frame
            + np.roll(frame, 1, axis=0)
            + np.roll(frame, -1, axis=0)
            + np.roll(frame, 1, axis=1)
            + np.roll(frame, -1, axis=1)
        )
        times.append(index / fps)
        means.append(float(frame.mean()))
        stds.append(float(frame.std()))
        blurs.append(float(laplacian.var()))
        diffs.append(
            0.0 if previous is None else float(np.abs(frame - previous).mean())
        )
        darks.append(float((frame < cfg.video_dark_luma).mean()))
        brights.append(float((frame > cfg.video_bright_luma).mean()))
        previous = frame
        index += 1

    returncode = process.wait()
    error = None if returncode == 0 else f"ffmpeg decode failed (rc={returncode})"
    return SampleSeries(
        times=np.asarray(times, dtype=np.float64),
        mean_luma=np.asarray(means, dtype=np.float64),
        std_luma=np.asarray(stds, dtype=np.float64),
        blur=np.asarray(blurs, dtype=np.float64),
        frame_diff=np.asarray(diffs, dtype=np.float64),
        dark_fraction=np.asarray(darks, dtype=np.float64),
        bright_fraction=np.asarray(brights, dtype=np.float64),
        fps=float(fps),
        decode_error=error,
    )


def _event_issue(
    issue_id: str,
    severity: str,
    ep: EpisodeMeta,
    camera: str,
    event: tuple[int, int],
    sample_indices: np.ndarray,
    times: np.ndarray,
    from_s: float,
    to_s: float,
    metric_text: str,
    event_count: int,
    fps: float,
) -> Issue:
    start_i, end_i = event
    start_time = float(times[sample_indices[start_i]])
    end_time = float(times[sample_indices[end_i]])
    start_frame = _to_frame(start_time, from_s, to_s, ep.length)
    end_frame = _to_frame(end_time, from_s, to_s, ep.length)
    duration = max(1, end_i - start_i + 1) / fps
    detail = (
        f"{camera}: {duration:.2f}s, frames {start_frame}-{end_frame}; {metric_text}"
        + (f"; {event_count} event(s) in this episode" if event_count > 1 else "")
    )
    return make_issue(
        issue_id,
        severity,
        ep.episode_index,
        start_frame,
        max(start_frame, end_frame),
        max(1, end_frame - start_frame + 1),
        detail,
    )


def analyze_episode(
    ep: EpisodeMeta,
    camera: str,
    series: SampleSeries,
    indices: np.ndarray,
    from_s: float,
    to_s: float,
    blur_threshold: float,
    cfg: Config,
) -> tuple[list[Issue], VideoQualityRow]:
    if indices.size == 0:
        return [], VideoQualityRow(
            ep.episode_index,
            camera,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            "ok",
            "",
        )

    mean = series.mean_luma[indices]
    std = series.std_luma[indices]
    blur = series.blur[indices]
    diff = series.frame_diff[indices].copy()
    diff[0] = np.nan  # first sample has no predecessor inside the episode
    bright = series.bright_fraction[indices]

    black = (mean < cfg.video_black_luma) & (std < cfg.video_black_std)
    dark = (mean < cfg.video_dark_luma) & (std < cfg.video_black_std) & ~black
    over = bright > cfg.video_bright_fraction
    frozen = np.nan_to_num(diff, nan=1.0) < cfg.video_freeze_diff
    blurry = (
        (blur < blur_threshold)
        & (std >= cfg.video_blur_min_texture_std)
        & ~black
        & ~frozen
    )

    fps = series.fps
    min_samples = max(1, int(math.ceil(cfg.video_event_min_duration_s * fps)))
    freeze_samples = max(1, int(math.ceil(cfg.video_freeze_min_duration_s * fps)))
    blur_samples = max(1, int(math.ceil(cfg.video_blur_min_duration_s * fps)))

    issues: list[Issue] = []
    event_specs = [
        ("l3.video_black", "fail", black, min_samples,
         f"black frames (mean luma<{cfg.video_black_luma:g}, std<{cfg.video_black_std:g})"),
        ("l3.video_dark", "warn", dark, min_samples,
         f"dark frames (mean luma<{cfg.video_dark_luma:g})"),
        ("l3.video_overexposed", "warn", over, min_samples,
         f"overexposed frames (bright fraction>{cfg.video_bright_fraction:g})"),
        ("l3.video_frozen", "warn", frozen, freeze_samples,
         f"frozen frames (frame diff<{cfg.video_freeze_diff:g})"),
        ("l3.video_blur", "warn", blurry, blur_samples,
         f"blurred frames (Laplacian variance<{blur_threshold:.1f})"),
    ]
    issue_ids: list[str] = []
    worst = "ok"
    for issue_id, severity, mask, minimum, label in event_specs:
        runs = _runs(mask, minimum)
        if not runs:
            continue
        best = max(runs, key=lambda run: run[1] - run[0])
        if issue_id == "l3.video_black":
            metric = f"mean luma min={float(np.nanmin(mean)):.1f}"
        elif issue_id == "l3.video_dark":
            metric = f"mean luma min={float(np.nanmin(mean)):.1f}"
        elif issue_id == "l3.video_overexposed":
            metric = f"bright fraction max={float(np.nanmax(bright)):.2f}"
        elif issue_id == "l3.video_frozen":
            metric = f"frame diff median={float(np.nanmedian(diff[best[0]:best[1]+1])):.3f}"
        else:
            metric = f"Laplacian variance min={float(np.nanmin(blur[best[0]:best[1]+1])):.1f}"
        issues.append(
            _event_issue(
                issue_id,
                severity,
                ep,
                camera,
                best,
                indices,
                series.times,
                from_s,
                to_s,
                f"{label}; {metric}",
                len(runs),
                series.fps,
            )
        )
        issue_ids.append(issue_id)
        if severity == "fail" or (severity == "warn" and worst == "ok"):
            worst = severity

    row = VideoQualityRow(
        episode_index=ep.episode_index,
        camera=camera,
        samples=int(indices.size),
        black_ratio=float(np.mean(black)),
        dark_ratio=float(np.mean(dark)),
        bright_ratio=float(np.mean(over)),
        freeze_ratio=float(np.mean(frozen)),
        blur_low_ratio=float(np.mean(blurry)),
        blur_min=float(np.nanmin(blur)),
        blur_median=float(np.nanmedian(blur)),
        luma_min=float(np.nanmin(mean)),
        luma_median=float(np.nanmedian(mean)),
        luma_max=float(np.nanmax(mean)),
        status=worst,
        issues=",".join(sorted(set(issue_ids))),
    )
    return issues, row


def check_video_quality(
    root: Path,
    info: dict,
    episodes: list[EpisodeMeta],
    cfg: Config,
    *,
    progress: ProgressBar | None = None,
) -> tuple[list[Issue], list[Issue], list[VideoQualityRow], dict]:
    """Return (dataset issues, episode issues, rows, calibration)."""
    dataset_issues: list[Issue] = []
    episode_issues: list[Issue] = []
    rows: list[VideoQualityRow] = []

    if not cfg.video_quality_enabled:
        return dataset_issues, episode_issues, rows, {"enabled": False}
    if shutil.which("ffmpeg") is None:
        dataset_issues.append(
            dataset_issue("l3.ffmpeg_missing", "warn", "ffmpeg not found; video frame quality skipped")
        )
        return dataset_issues, episode_issues, rows, {"enabled": False, "reason": "ffmpeg missing"}

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
        return dataset_issues, episode_issues, rows, {"enabled": True, "videos": 0}

    series_by_file: dict[Path, SampleSeries] = {}
    fps_by_camera: dict[str, float] = {}
    for path in sorted(usages, key=str):
        probe = probe_video(path)
        native_fps = probe.fps if probe.fps and probe.fps > 0 else float(info.get("fps", cfg.fps))
        series = extract_samples(path, cfg, native_fps)
        series_by_file[path] = series
        camera = usages[path][0][1]
        fps_by_camera.setdefault(camera, series.fps)
        if series.decode_error:
            dataset_issues.append(
                dataset_issue("l3.video_decode", "fail", f"{path.name}: {series.decode_error}")
            )
        if progress is not None:
            progress.update()

    # Per-camera sharpness baseline. The threshold is relative to the camera's
    # own median because the four cameras have very different depth of field.
    camera_blur: dict[str, list[np.ndarray]] = {}
    for path, series in series_by_file.items():
        for _ep, camera, _from, _to in usages[path]:
            camera_blur.setdefault(camera, []).append(series.blur)
            break
    calibration: dict = {
        "enabled": True,
        "full_frame": cfg.video_quality_full_frame,
        "sample_fps": None if cfg.video_quality_full_frame else cfg.video_quality_sample_fps,
        "frame_width": cfg.video_quality_width,
        "frame_height": cfg.video_quality_height,
        "cameras": {},
    }
    blur_thresholds: dict[str, float] = {}
    for camera, chunks in sorted(camera_blur.items()):
        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        median = float(np.median(values))
        p1, p5, p50, p95 = (
            float(np.percentile(values, q)) for q in (1, 5, 50, 95)
        )
        threshold = max(1.0, median * cfg.video_blur_ratio_threshold)
        blur_thresholds[camera] = threshold
        calibration["cameras"][camera] = {
            "fps": round(float(fps_by_camera.get(camera, cfg.fps)), 4),
            "blur_p1": round(p1, 3),
            "blur_p5": round(p5, 3),
            "blur_p50": round(p50, 3),
            "blur_p95": round(p95, 3),
            "blur_threshold": round(threshold, 3),
        }

    for path, uses in sorted(usages.items(), key=lambda item: str(item[0])):
        series = series_by_file[path]
        if series.times.size == 0:
            continue
        for ep, camera, from_s, to_s in uses:
            indices = np.where((series.times >= from_s) & (series.times < to_s))[0]
            threshold = blur_thresholds.get(
                camera, max(1.0, float(np.nanmedian(series.blur)) * cfg.video_blur_ratio_threshold)
            )
            issues, row = analyze_episode(
                ep, camera, series, indices, from_s, to_s, threshold, cfg
            )
            episode_issues.extend(issues)
            rows.append(row)

    flagged = [row for row in rows if row.status != "ok"]
    mode = (
        "full-frame"
        if cfg.video_quality_full_frame
        else f"{cfg.video_quality_sample_fps:g} fps"
    )
    dataset_issues.append(
        dataset_issue(
            "l3.video_quality_baseline",
            "info",
            f"processed {len(series_by_file)} video file(s), {mode}, "
            f"{cfg.video_quality_width}x{cfg.video_quality_height} gray; "
            f"{len(flagged)} episode/camera row(s) flagged",
        )
    )
    return dataset_issues, episode_issues, rows, calibration
