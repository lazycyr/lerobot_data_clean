"""Trajectory-level deviation detection and threshold calibration.

Each episode's action trajectory is compared with the dataset-wide median
trajectory per frame and dimension. Episodes of different lengths are compared
on the common frame axis with NaN padding: each episode is scored only over its
own frames, and frames with too few episodes are skipped. The per-frame
deviation is robustly standardized (median/MAD), aggregated across continuous
dimensions, smoothed, and scanned for sustained runs. This catches "this
episode takes a different path for a while", as opposed to the instantaneous
joint-step check.

The detector also returns per-episode metrics for every episode, including
clean ones, so a new dataset can be calibrated from its own score distribution
instead of reusing an arbitrary threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .dataset import EpisodeArrays
from .scoring import Issue, dataset_issue, make_issue


@dataclass(frozen=True)
class TrajectoryEpisodeMetric:
    episode_index: int
    computed: bool
    max_smoothed_score: float | None = None
    max_score_frame: int | None = None
    max_abs_offset: float | None = None
    max_abs_frame: int | None = None
    worst_dim: int | None = None
    worst_dim_name: str | None = None
    segment_start: int | None = None
    segment_end: int | None = None
    flagged: bool = False
    severity: str | None = None
    note: str = ""


@dataclass
class TrajectoryResult:
    issues: list[Issue]
    dataset_issue: Issue | None
    metrics: list[TrajectoryEpisodeMetric]


def _dim_name(info: dict, feature: str, index: int) -> str:
    key = "observation.state" if feature == "state" else feature
    names = info.get("features", {}).get(key, {}).get("names") or []
    if 0 <= index < len(names):
        return str(names[index])
    return f"dim_{index}"


def _moving_average_rows(
    values: np.ndarray,
    lengths: list[int],
    window: int,
) -> np.ndarray:
    """Moving average along axis 1, only over each row's valid prefix.

    Rows are NaN-padded to a common length; the padded tail stays NaN so it
    cannot contribute to runs or argmax. Edge padding is used at both ends.
    """
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.full_like(values, np.nan)
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        padded = np.pad(values[row, :length], (pad, pad), mode="edge")
        out[row, :length] = np.convolve(padded, kernel, mode="valid")
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[list[int]] = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(int(start), int(end)) for start, end in merged]


def calibrate_trajectory(
    metrics: list[TrajectoryEpisodeMetric],
    current_threshold: float,
) -> dict:
    """Summarize the score distribution and suggest a threshold.

    The suggestion looks for the largest gap in the upper half of the sorted
    max-score distribution. If there is no clear gap, it falls back to p99.
    A new dataset should inspect this JSON before reusing a threshold.
    """
    computed = [
        metric
        for metric in metrics
        if metric.computed and metric.max_smoothed_score is not None
    ]
    scores = sorted(float(metric.max_smoothed_score) for metric in computed)
    result: dict = {
        "n_episodes": len(metrics),
        "computed_episodes": len(computed),
        "current_threshold": current_threshold,
        "quantiles": {},
        "largest_gap": None,
        "suggested_threshold": None,
        "flagged_episodes": [m.episode_index for m in computed if m.flagged],
        "top_scores": [
            {
                "episode_index": m.episode_index,
                "max_smoothed_score": round(float(m.max_smoothed_score), 4),
                "max_abs_offset": (
                    round(float(m.max_abs_offset), 6)
                    if m.max_abs_offset is not None
                    else None
                ),
                "segment_start": m.segment_start,
                "segment_end": m.segment_end,
                "flagged": m.flagged,
            }
            for m in sorted(computed, key=lambda item: -float(item.max_smoothed_score))[:20]
        ],
    }
    if not scores:
        result["note"] = "no computed episodes; trajectory check was skipped"
        return result

    result["quantiles"] = {
        f"p{q}": round(float(np.percentile(scores, q)), 4)
        for q in (50, 90, 95, 97, 98, 99, 100)
    }

    median = float(np.median(scores))
    largest_gap: dict | None = None
    for low, high in zip(scores, scores[1:]):
        if low < median:
            continue
        gap = high - low
        if largest_gap is None or gap > largest_gap["gap"]:
            largest_gap = {
                "low": round(low, 4),
                "high": round(high, 4),
                "gap": round(gap, 4),
                "midpoint": round((low + high) / 2.0, 4),
            }
    result["largest_gap"] = largest_gap

    if largest_gap is not None and largest_gap["gap"] >= max(2.0, 0.25 * median):
        suggested = largest_gap["midpoint"]
    else:
        suggested = float(np.percentile(scores, 99))
    result["suggested_threshold"] = round(float(suggested), 4)
    return result


def check_trajectory_deviation(
    data: dict[int, EpisodeArrays],
    info: dict,
    cfg: Config,
) -> TrajectoryResult:
    """Return per-episode issues, a dataset baseline/info issue, and metrics."""
    all_episode_ids = sorted(int(episode) for episode in data)
    issues: list[Issue] = []

    usable = {
        int(episode): arrays
        for episode, arrays in data.items()
        if arrays.num_rows >= cfg.trajectory_min_segment_frames
        and np.isfinite(arrays.action).all()
    }
    if len(usable) < cfg.trajectory_min_episodes:
        reason = (
            f"need at least {cfg.trajectory_min_episodes} finite episodes; "
            f"got {len(usable)}"
        )
        metrics = [
            TrajectoryEpisodeMetric(episode, False, note=reason)
            for episode in all_episode_ids
        ]
        return TrajectoryResult(
            issues=issues,
            dataset_issue=dataset_issue("l2.trajectory_not_computed", "info", reason),
            metrics=metrics,
        )

    episode_ids = sorted(usable)
    lengths = [usable[ep].num_rows for ep in episode_ids]
    max_length = max(lengths)

    lo, hi = cfg.trajectory_dims
    max_dim = min(arrays.action.shape[1] for arrays in usable.values()) - 1
    hi = min(hi, max_dim)
    if hi < lo:
        reason = f"trajectory dims {cfg.trajectory_dims} outside available action dimensions"
        metrics = [
            TrajectoryEpisodeMetric(episode, False, note=reason)
            for episode in all_episode_ids
        ]
        return TrajectoryResult(
            issues=issues,
            dataset_issue=dataset_issue("l2.trajectory_not_computed", "info", reason),
            metrics=metrics,
        )
    dims = list(range(lo, hi + 1))

    # NaN-pad to the longest episode so unequal lengths share one baseline.
    stacked = np.full((len(episode_ids), max_length, len(dims)), np.nan)
    for row, ep in enumerate(episode_ids):
        stacked[row, : lengths[row]] = usable[ep].action[:, dims]

    median = np.nanmedian(stacked, axis=0)
    mad = np.nanmedian(np.abs(stacked - median), axis=0)
    scale = np.maximum(1.4826 * mad, cfg.trajectory_scale_floor)
    # Skip frames where too few episodes are alive to form a stable baseline.
    valid_per_frame = np.isfinite(stacked[:, :, 0]).sum(axis=0)
    enough = valid_per_frame >= cfg.trajectory_min_episodes
    median[~enough] = np.nan
    scale[~enough] = np.nan

    z = np.abs(stacked - median) / scale
    finite = np.isfinite(z)
    z2_sum = np.where(finite, z * z, 0.0).sum(axis=2)
    z_count = finite.sum(axis=2)
    score = np.sqrt(z2_sum / np.maximum(z_count, 1))
    score[z_count == 0] = np.nan
    smoothed = _moving_average_rows(score, lengths, cfg.trajectory_smooth_frames)
    abs_residual_all = np.abs(stacked - median)

    metrics_by_episode: dict[int, TrajectoryEpisodeMetric] = {}
    flagged = 0
    for row, episode in enumerate(episode_ids):
        segment_scores = smoothed[row]
        rel_score_frame = int(np.nanargmax(segment_scores))
        max_smoothed_score = float(segment_scores[rel_score_frame])

        flat_all = int(np.nanargmax(abs_residual_all[row]))
        abs_frame, abs_dim = np.unravel_index(flat_all, abs_residual_all[row].shape)
        max_abs_offset = float(abs_residual_all[row, abs_frame, abs_dim])

        mask = segment_scores > cfg.trajectory_z_warn
        runs = _merge_runs(_runs(mask), cfg.trajectory_merge_gap_frames)
        candidates: list[tuple[float, float, int, int]] = []
        for start, end in runs:
            if end - start + 1 < cfg.trajectory_min_segment_frames:
                continue
            abs_residual = abs_residual_all[row, start : end + 1]
            if abs_residual.size == 0:
                continue
            segment_max_abs = float(np.nanmax(abs_residual))
            if segment_max_abs < cfg.trajectory_abs_floor:
                continue
            segment_max_score = float(np.nanmax(segment_scores[start : end + 1]))
            candidates.append((segment_max_score, segment_max_abs, start, end))

        if not candidates:
            metrics_by_episode[episode] = TrajectoryEpisodeMetric(
                episode_index=episode,
                computed=True,
                max_smoothed_score=max_smoothed_score,
                max_score_frame=rel_score_frame,
                max_abs_offset=max_abs_offset,
                max_abs_frame=int(abs_frame),
                worst_dim=int(dims[int(abs_dim)]),
                worst_dim_name=_dim_name(info, "action", int(dims[int(abs_dim)])),
                flagged=False,
                note="below trajectory threshold",
            )
            continue

        flagged += 1
        _, _, start, end = max(candidates, key=lambda item: (item[0], item[1]))
        seg_scores = smoothed[row, start : end + 1]
        rel_seg_score = int(np.argmax(seg_scores))
        max_score = float(seg_scores[rel_seg_score])
        score_frame = start + rel_seg_score

        abs_segment = abs_residual_all[row, start : end + 1]
        rel_abs, rel_dim = np.unravel_index(int(np.nanargmax(abs_segment)), abs_segment.shape)
        worst_frame = start + int(rel_abs)
        worst_dim = int(dims[int(rel_dim)])
        segment_max_abs = float(abs_segment.flat[int(np.nanargmax(abs_segment))])
        severity = (
            "fail"
            if max_score >= cfg.trajectory_z_fail or segment_max_abs >= cfg.trajectory_abs_fail
            else "warn"
        )
        other = len(candidates) - 1
        detail = (
            f"deviates from dataset median trajectory for {end - start + 1} frames; "
            f"max score={max_score:.1f} at frame {score_frame}, "
            f"max |Δ|={segment_max_abs:.3f} at frame {worst_frame} on "
            f"{_dim_name(info, 'action', worst_dim)}"
        )
        if other:
            detail += f"; {other} other deviating segment(s) in this episode"
        issues.append(
            make_issue(
                "l2.trajectory_deviation",
                severity,
                episode,
                start,
                end,
                end - start + 1,
                detail,
            )
        )
        metrics_by_episode[episode] = TrajectoryEpisodeMetric(
            episode_index=episode,
            computed=True,
            max_smoothed_score=max_smoothed_score,
            max_score_frame=rel_score_frame,
            max_abs_offset=max_abs_offset,
            max_abs_frame=int(abs_frame),
            worst_dim=int(dims[int(abs_dim)]),
            worst_dim_name=_dim_name(info, "action", int(dims[int(abs_dim)])),
            segment_start=start,
            segment_end=end,
            flagged=True,
            severity=severity,
        )

    metrics: list[TrajectoryEpisodeMetric] = []
    for episode in all_episode_ids:
        if episode in metrics_by_episode:
            metrics.append(metrics_by_episode[episode])
        else:
            metrics.append(
                TrajectoryEpisodeMetric(
                    episode_index=episode,
                    computed=False,
                    note="non-finite action or too short for the trajectory check",
                )
            )

    baseline = dataset_issue(
        "l2.trajectory_baseline",
        "info",
        f"median action trajectory over {len(usable)} episodes "
        f"(lengths {min(lengths)}-{max_length}, NaN-padded), "
        f"dims {lo}-{hi}; {flagged} episode(s) exceed the deviation threshold",
    )
    return TrajectoryResult(issues=issues, dataset_issue=baseline, metrics=metrics)
