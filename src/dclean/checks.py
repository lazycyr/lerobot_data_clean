"""Layer checks: L0 metadata/inventory, L1 time, L2 numeric.

Only ``state[:, :20]`` is read by the numeric checks; object-pose dimensions
20..47 are deliberately ignored.
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config
from .dataset import (
    DEFAULT_DATA_COLUMNS,
    EpisodeArrays,
    EpisodeMeta,
    group_episodes_by_data_file,
    resolve_data_path,
    resolve_video_path,
    video_cameras,
)
from .scoring import (
    Issue,
    dataset_issue,
    examples_text,
    frame_span,
    make_issue,
    transition_frames,
)


# ---------------------------------------------------------------------------
# helpers


_FEATURE_KEYS = {"state": "observation.state"}


def _names(info: dict, feature: str) -> list[str]:
    key = _FEATURE_KEYS.get(feature, feature)
    spec = info.get("features", {}).get(key, {})
    names = spec.get("names") or []
    return [str(name) for name in names]


def _dim_name(info: dict, feature: str, index: int) -> str:
    names = _names(info, feature)
    if 0 <= index < len(names):
        return names[index]
    return f"dim_{index}"


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _numbering_issues(root: Path, paths: list[Path], pattern: str, label: str) -> list[Issue]:
    issues: list[Issue] = []
    bad_names: list[str] = []
    by_dir: dict[Path, list[int]] = {}
    regex = re.compile(pattern)
    for path in paths:
        match = regex.fullmatch(path.name)
        if not match:
            bad_names.append(_rel(root, path))
            continue
        by_dir.setdefault(path.parent, []).append(int(match.group(1)))

    if bad_names:
        issues.append(
            dataset_issue(
                "l0.file_naming",
                "warn",
                f"{label}: {len(bad_names)} file name(s) do not match the v3.0 pattern; "
                f"examples: {', '.join(bad_names[:5])}",
            )
        )

    gap_details: list[str] = []
    for directory, numbers in sorted(by_dir.items(), key=lambda item: str(item[0])):
        numbers = sorted(numbers)
        expected = list(range(numbers[0], numbers[-1] + 1))
        missing = sorted(set(expected) - set(numbers))
        if missing:
            shown = ", ".join(str(n) for n in missing[:5])
            suffix = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5})"
            gap_details.append(f"{_rel(root, directory)}: missing file indices {shown}{suffix}")
    if gap_details:
        issues.append(
            dataset_issue(
                "l0.file_numbering",
                "warn",
                f"{label}: non-contiguous file numbering; " + "; ".join(gap_details[:5]),
            )
        )
    return issues


# ---------------------------------------------------------------------------
# L0 metadata


def check_metadata_totals(
    root: Path,
    info: dict,
    tasks: dict[int, str],
    episodes: list[EpisodeMeta],
) -> list[Issue]:
    issues: list[Issue] = []
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    total_tasks = int(info["total_tasks"])

    indices = [ep.episode_index for ep in episodes]
    expected_indices = list(range(total_episodes))
    if indices != expected_indices:
        counts = Counter(indices)
        missing = sorted(set(expected_indices) - set(indices))
        extra = sorted(set(indices) - set(expected_indices))
        duplicates = sorted(i for i, c in counts.items() if c > 1)
        parts = []
        if missing:
            parts.append(f"missing {missing[:8]}")
        if extra:
            parts.append(f"extra {extra[:8]}")
        if duplicates:
            parts.append(f"duplicated {duplicates[:8]}")
        issues.append(
            dataset_issue(
                "l0.episode_index",
                "fail",
                "episode_index is not a contiguous 0..N-1 sequence: " + "; ".join(parts),
            )
        )

    if len(episodes) != total_episodes:
        issues.append(
            dataset_issue(
                "l0.metadata_totals",
                "fail",
                f"metadata has {len(episodes)} episodes, info.json declares {total_episodes}",
            )
        )

    frame_sum = sum(ep.length for ep in episodes)
    if frame_sum != total_frames:
        issues.append(
            dataset_issue(
                "l0.metadata_totals",
                "fail",
                f"sum(length)={frame_sum}, info.json total_frames={total_frames}",
            )
        )

    if len(tasks) != total_tasks:
        issues.append(
            dataset_issue(
                "l0.metadata_totals",
                "fail",
                f"tasks.parquet has {len(tasks)} tasks, info.json declares {total_tasks}",
            )
        )

    offset_problems: list[str] = []
    bad_length_eps: list[int] = []
    if not episodes:
        offset_problems.append("no episodes")
    else:
        if episodes[0].dataset_from_index != 0:
            offset_problems.append(f"first dataset_from_index={episodes[0].dataset_from_index}, expected 0")
        if episodes[-1].dataset_to_index != total_frames:
            offset_problems.append(
                f"last dataset_to_index={episodes[-1].dataset_to_index}, expected {total_frames}"
            )
        for ep in episodes:
            if ep.dataset_to_index - ep.dataset_from_index != ep.length:
                bad_length_eps.append(ep.episode_index)
        if bad_length_eps:
            offset_problems.append(f"{len(bad_length_eps)} episodes have to-from != length")
        jumps = []
        for left, right in zip(episodes, episodes[1:]):
            if left.dataset_to_index != right.dataset_from_index:
                jumps.append((left.episode_index, right.episode_index))
        if jumps:
            offset_problems.append(f"{len(jumps)} adjacent offset gaps/overlaps")
    if offset_problems:
        issues.append(
            dataset_issue(
                "l0.global_offset",
                "fail",
                "invalid dataset offsets: " + "; ".join(offset_problems),
            )
        )

    # Data schema from the first existing data file.
    first_path = next(
        (resolve_data_path(root, info, ep) for ep in episodes if resolve_data_path(root, info, ep).is_file()),
        None,
    )
    if first_path is None:
        issues.append(
            dataset_issue("l0.data_schema", "fail", "cannot validate schema: no data file exists")
        )
    else:
        try:
            schema = pq.ParquetFile(first_path).schema_arrow
        except Exception as exc:  # noqa: BLE001 - report, do not crash the check
            issues.append(
                dataset_issue("l0.data_schema", "fail", f"cannot read {_rel(root, first_path)}: {exc}")
            )
        else:
            missing = sorted(set(DEFAULT_DATA_COLUMNS) - set(schema.names))
            if missing:
                issues.append(
                    dataset_issue(
                        "l0.data_schema",
                        "fail",
                        f"{_rel(root, first_path)} missing columns: {missing}",
                    )
                )
            action_shape = info.get("features", {}).get("action", {}).get("shape") or [20]
            state_shape = info.get("features", {}).get("observation.state", {}).get("shape") or [48]
            for feature, dim in (("action", int(action_shape[0])), ("observation.state", int(state_shape[0]))):
                if feature not in schema.names:
                    continue
                field_type = schema.field(feature).type
                if not pa.types.is_fixed_size_list(field_type) or field_type.list_size != dim:
                    issues.append(
                        dataset_issue(
                            "l0.data_schema",
                            "fail",
                            f"{feature} expected fixed_size_list[{dim}], got {field_type}",
                        )
                    )

    # Per-file row counts.
    groups = group_episodes_by_data_file(root, info, episodes)
    row_problems: list[str] = []
    for path, eps in sorted(groups.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            continue
        try:
            rows = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:  # noqa: BLE001
            row_problems.append(f"{_rel(root, path)}: unreadable ({exc})")
            continue
        expected_rows = sum(ep.length for ep in eps)
        if rows != expected_rows:
            ep_min = min(ep.episode_index for ep in eps)
            ep_max = max(ep.episode_index for ep in eps)
            row_problems.append(
                f"{_rel(root, path)}: {rows} rows, expected {expected_rows} "
                f"(episodes {ep_min}-{ep_max})"
            )
    if row_problems:
        issues.append(
            dataset_issue(
                "l0.file_row_count",
                "fail",
                f"{len(row_problems)} data file(s) have row-count mismatch; "
                + "; ".join(row_problems[:5]),
            )
        )

    # Cheap stats-count reconciliation (never trust the stats values themselves).
    length_by_episode = {ep.episode_index: ep.length for ep in episodes}
    stat_mismatches: list[str] = []
    metadata_paths = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
    for path in metadata_paths:
        try:
            pf = pq.ParquetFile(path)
            count_cols = [
                col
                for col in ("stats/action/count", "stats/observation.state/count")
                if col in pf.schema_arrow.names
            ]
            if not count_cols:
                continue
            table = pf.read(columns=["episode_index", *count_cols])
        except Exception:  # noqa: BLE001 - a broken stats column is not fatal
            continue
        for row in table.to_pylist():
            ep_index = int(row["episode_index"])
            expected = length_by_episode.get(ep_index)
            if expected is None:
                continue
            for col in count_cols:
                value = row[col]
                if isinstance(value, list):
                    value = value[0] if value else None
                if value is not None and int(value) != expected:
                    stat_mismatches.append(f"ep{ep_index} {col}={value}, expected {expected}")
    if stat_mismatches:
        issues.append(
            dataset_issue(
                "l0.stats_count",
                "warn",
                f"{len(stat_mismatches)} stats count mismatch(es); "
                + "; ".join(stat_mismatches[:5]),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# L0 inventory


def check_inventory(
    root: Path,
    info: dict,
    episodes: list[EpisodeMeta],
    *,
    selected_indices: set[int] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []

    required = ["meta/info.json", "meta/tasks.parquet", "meta/episodes", "data", "videos"]
    for rel in required:
        if not (root / rel).exists():
            issues.append(
                dataset_issue("l0.missing_required_path", "fail", f"missing required path: {rel}")
            )
    if not (root / "meta" / "stats.json").is_file():
        issues.append(
            dataset_issue("l0.missing_required_path", "warn", "missing optional metadata: meta/stats.json")
        )
    for camera in video_cameras(info):
        rel = f"videos/{camera}"
        if not (root / rel).is_dir():
            issues.append(
                dataset_issue("l0.missing_required_path", "fail", f"missing camera directory: {rel}")
            )

    # Referenced files, grouped per episode so the report stays readable.
    for ep in episodes:
        if selected_indices is not None and ep.episode_index not in selected_indices:
            continue
        missing: list[str] = []
        data_path = resolve_data_path(root, info, ep)
        if not _nonempty_file(data_path):
            missing.append(f"data:{_rel(root, data_path)}")
        for camera in ep.videos:
            video_path = resolve_video_path(root, info, ep, camera)
            if not _nonempty_file(video_path):
                missing.append(f"video:{camera}:{_rel(root, video_path)}")
        if missing:
            issues.append(
                make_issue(
                    "l0.missing_referenced_file",
                    "fail",
                    ep.episode_index,
                    0,
                    max(0, ep.length - 1),
                    len(missing),
                    "missing or empty referenced file(s): " + "; ".join(missing[:5]),
                )
            )

    referenced_data = {str(resolve_data_path(root, info, ep).resolve()) for ep in episodes}
    referenced_video = {
        str(resolve_video_path(root, info, ep, camera).resolve())
        for ep in episodes
        for camera in ep.videos
    }
    all_data = {str(Path(p).resolve()) for p in glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet"))}
    all_video = {
        str(Path(p).resolve())
        for p in glob.glob(str(root / "videos" / "*" / "chunk-*" / "file-*.mp4"))
    }

    orphans = sorted((all_data - referenced_data) | (all_video - referenced_video))
    if orphans:
        shown = [_rel(root, Path(p)) for p in orphans[:8]]
        suffix = "" if len(orphans) <= 8 else f" ... (+{len(orphans) - 8})"
        issues.append(
            dataset_issue(
                "l0.orphan_file",
                "warn",
                f"{len(orphans)} unreferenced data/video file(s): " + ", ".join(shown) + suffix,
            )
        )

    empty = sorted(
        path
        for path in (all_data | all_video)
        if Path(path).is_file() and Path(path).stat().st_size == 0
    )
    if empty:
        shown = [_rel(root, Path(p)) for p in empty[:8]]
        issues.append(
            dataset_issue("l0.empty_file", "fail", f"{len(empty)} empty file(s): " + ", ".join(shown))
        )

    issues.extend(
        _numbering_issues(
            root,
            [Path(p) for p in all_data],
            r"file-(\d{3})\.parquet",
            "data",
        )
    )
    issues.extend(
        _numbering_issues(
            root,
            [Path(p) for p in all_video],
            r"file-(\d{3})\.mp4",
            "videos",
        )
    )

    # meta/episodes self-reference columns must match the actual path.
    meta_paths = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
    bad_self_ref: list[str] = []
    for path in meta_paths:
        chunk_match = re.fullmatch(r"chunk-(\d{3})", Path(path).parent.name)
        file_match = re.fullmatch(r"file-(\d{3})\.parquet", Path(path).name)
        if not chunk_match or not file_match:
            bad_self_ref.append(_rel(root, Path(path)))
            continue
        expected_chunk, expected_file = int(chunk_match.group(1)), int(file_match.group(1))
        try:
            pf = pq.ParquetFile(path)
            if (
                "meta/episodes/chunk_index" not in pf.schema_arrow.names
                or "meta/episodes/file_index" not in pf.schema_arrow.names
            ):
                issues.append(
                    dataset_issue(
                        "l0.episodes_meta_path",
                        "warn",
                        f"{_rel(root, Path(path))} has no meta/episodes/* self-reference columns",
                    )
                )
                continue
            table = pf.read(columns=["meta/episodes/chunk_index", "meta/episodes/file_index"])
        except Exception as exc:  # noqa: BLE001
            bad_self_ref.append(f"{_rel(root, Path(path))} unreadable ({exc})")
            continue
        for row in table.to_pylist():
            if (
                int(row["meta/episodes/chunk_index"]) != expected_chunk
                or int(row["meta/episodes/file_index"]) != expected_file
            ):
                bad_self_ref.append(_rel(root, Path(path)))
                break
    if bad_self_ref:
        issues.append(
            dataset_issue(
                "l0.episodes_meta_path",
                "fail",
                "meta/episodes self-reference mismatch: " + ", ".join(sorted(set(bad_self_ref))[:8]),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# L0 per-episode structure


def check_episode_structure(
    ep: EpisodeMeta,
    arrays: EpisodeArrays,
    tasks: dict[int, str],
) -> list[Issue]:
    issues: list[Issue] = []
    n = arrays.num_rows
    if n != ep.length:
        issues.append(
            make_issue(
                "l0.data_row_count",
                "fail",
                ep.episode_index,
                0,
                max(0, ep.length - 1, n - 1),
                1,
                f"expected {ep.length} rows, got {n}",
            )
        )
    if n == 0:
        return issues

    m = min(n, ep.length)

    if not np.all(arrays.episode_index[:m] == ep.episode_index):
        issues.append(
            make_issue(
                "l0.episode_index",
                "fail",
                ep.episode_index,
                0,
                m - 1,
                1,
                f"episode_index column is not constant {ep.episode_index}",
            )
        )

    expected_frame = np.arange(m, dtype=np.int64)
    bad_frame = np.where(arrays.frame_index[:m] != expected_frame)[0]
    if bad_frame.size:
        start, end, count = frame_span(bad_frame, ep.length) or (0, m - 1, int(bad_frame.size))
        issues.append(
            make_issue(
                "l0.frame_index",
                "fail",
                ep.episode_index,
                start,
                end,
                int(bad_frame.size),
                f"frame_index mismatch at {count} frame(s), examples: "
                f"{examples_text(bad_frame, 3)}",
            )
        )

    expected_index = ep.dataset_from_index + np.arange(m, dtype=np.int64)
    bad_index = np.where(arrays.index[:m] != expected_index)[0]
    if bad_index.size:
        start, end, _ = frame_span(bad_index, ep.length) or (0, m - 1, int(bad_index.size))
        issues.append(
            make_issue(
                "l0.global_index",
                "fail",
                ep.episode_index,
                start,
                end,
                int(bad_index.size),
                f"global index mismatch at {bad_index.size} frame(s); expected "
                f"{expected_index[0]}..{expected_index[-1]}, got "
                f"{int(arrays.index[0])}..{int(arrays.index[m - 1])}",
            )
        )

    if not np.all(arrays.task_index[:m] == arrays.task_index[0]):
        issues.append(
            make_issue(
                "l0.task_mapping",
                "fail",
                ep.episode_index,
                0,
                m - 1,
                1,
                "task_index is not constant within the episode",
            )
        )
    elif int(arrays.task_index[0]) not in tasks:
        issues.append(
            make_issue(
                "l0.task_mapping",
                "fail",
                ep.episode_index,
                0,
                m - 1,
                1,
                f"task_index {int(arrays.task_index[0])} not present in meta/tasks.parquet",
            )
        )

    if ep.dataset_to_index - ep.dataset_from_index != ep.length:
        issues.append(
            make_issue(
                "l0.global_offset",
                "fail",
                ep.episode_index,
                0,
                max(0, ep.length - 1),
                1,
                f"dataset_to_index - dataset_from_index = "
                f"{ep.dataset_to_index - ep.dataset_from_index}, length = {ep.length}",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# L1 time


def check_timestamps(ep: EpisodeMeta, arrays: EpisodeArrays, cfg: Config) -> list[Issue]:
    ts = arrays.timestamp
    n = len(ts)
    if n < 2:
        return []
    expected_dt = 1.0 / cfg.fps
    dt = np.diff(ts)
    issues: list[Issue] = []

    def add(issue_id: str, severity: str, transitions: np.ndarray, detail: str) -> None:
        frames = transition_frames(transitions.tolist(), ep.length)
        span = frame_span(frames, ep.length)
        if span is None:
            return
        start, end, _ = span
        issues.append(
            make_issue(
                issue_id,
                severity,
                ep.episode_index,
                start,
                end,
                int(transitions.size),
                detail,
            )
        )

    nonmono = np.where(dt < -1e-12)[0]
    if nonmono.size:
        worst = float(np.min(dt))
        add(
            "l1.timestamp_nonmonotonic",
            "fail",
            nonmono,
            f"{nonmono.size} non-monotonic transition(s), min dt={worst:.6f}s; "
            f"examples: {examples_text(nonmono, cfg.max_examples_in_detail)}",
        )

    duplicate = np.where(np.abs(dt) <= 1e-9)[0]
    if duplicate.size:
        add(
            "l1.timestamp_duplicate",
            "fail",
            duplicate,
            f"{duplicate.size} zero-dt transition(s); examples: "
            f"{examples_text(duplicate, cfg.max_examples_in_detail)}",
        )

    gap_threshold = expected_dt * (1.0 + 0.5 * max(1, cfg.max_gap_frames))
    gap = np.where(dt > gap_threshold)[0]
    if gap.size:
        add(
            "l1.timestamp_gap",
            "warn",
            gap,
            f"{gap.size} gap(s) larger than {gap_threshold:.6f}s (max dt={float(np.max(dt[gap])):.6f}s); "
            f"examples: {examples_text(gap, cfg.max_examples_in_detail)}",
        )

    excluded = np.zeros_like(dt, dtype=bool)
    for arr in (nonmono, duplicate, gap):
        excluded[arr] = True
    outlier = np.where((np.abs(dt - expected_dt) > cfg.dt_abs_tol_s) & ~excluded)[0]
    if outlier.size:
        add(
            "l1.timestamp_dt_outlier",
            "warn",
            outlier,
            f"{outlier.size} dt outlier(s) outside {cfg.dt_abs_tol_s}s of {expected_dt:.6f}s; "
            f"examples: {examples_text(outlier, cfg.max_examples_in_detail)}",
        )

    return issues


# ---------------------------------------------------------------------------
# L2 numeric


def check_finite(ep: EpisodeMeta, arrays: EpisodeArrays) -> Issue | None:
    action = arrays.action
    state = arrays.state
    bad = np.isnan(action) | np.isinf(action) | np.isnan(state) | np.isinf(state)
    frames = np.where(bad.any(axis=1))[0]
    if frames.size == 0:
        return None
    start, end, _ = frame_span(frames, ep.length) or (0, int(frames[-1]), int(frames.size))
    return make_issue(
        "l2.nan_inf",
        "fail",
        ep.episode_index,
        start,
        end,
        int(frames.size),
        f"{int(np.isnan(action).sum())} NaN / {int(np.isinf(action).sum())} Inf in action, "
        f"{int(np.isnan(state).sum())} NaN / {int(np.isinf(state).sum())} Inf in state[:20]; "
        f"frames: {examples_text(frames, 3)}",
    )


def global_dim_stats(
    data: dict[int, EpisodeArrays],
    feature: str,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (global_std, per-episode constant fraction, global_constant_mask)."""
    matrices = [getattr(arrays, feature) for arrays in data.values() if arrays.num_rows > 0]
    if not matrices:
        empty = np.zeros(0)
        return empty, empty, np.zeros(0, dtype=bool)
    dim = matrices[0].shape[1]
    matrices = [m[:, :dim] for m in matrices]
    pooled = np.concatenate(matrices, axis=0)
    pooled = np.where(np.isfinite(pooled), pooled, np.nan)
    global_std = np.nanstd(pooled, axis=0)
    constant = np.zeros((len(matrices), dim), dtype=bool)
    for i, matrix in enumerate(matrices):
        finite_matrix = np.where(np.isfinite(matrix), matrix, np.nan)
        constant[i] = np.nanstd(finite_matrix, axis=0) <= cfg.constant_std_eps
    fraction = constant.mean(axis=0) if len(matrices) else np.zeros(dim)
    global_constant = fraction >= cfg.constant_global_fraction
    return global_std, fraction, global_constant


def global_constant_issue(
    info: dict,
    feature: str,
    global_constant: np.ndarray,
) -> Issue | None:
    dims = np.where(global_constant)[0]
    if dims.size == 0:
        return None
    labels = [f"{feature}[{int(i)}]{_dim_name(info, feature, int(i))}" for i in dims]
    return dataset_issue(
        "l2.constant_channel_global",
        "info",
        f"{feature}: {len(labels)} globally constant channel(s): " + ", ".join(labels[:8]),
    )


def check_constant_channels(
    ep: EpisodeMeta,
    arrays: EpisodeArrays,
    info: dict,
    global_std_action: np.ndarray,
    global_constant_action: np.ndarray,
    global_std_state: np.ndarray,
    global_constant_state: np.ndarray,
    cfg: Config,
) -> Issue | None:
    labels: list[str] = []
    for feature, matrix, global_std, global_constant in (
        ("action", arrays.action, global_std_action, global_constant_action),
        ("state", arrays.state, global_std_state, global_constant_state),
    ):
        if matrix.size == 0:
            continue
        dim = matrix.shape[1]
        finite_matrix = np.where(np.isfinite(matrix), matrix, np.nan)
        std = np.nanstd(finite_matrix, axis=0)
        mask = (
            (std <= cfg.constant_std_eps)
            & (global_std[:dim] > 1e-3)
            & (~global_constant[:dim])
        )
        for idx in np.where(mask)[0]:
            labels.append(f"{feature}[{int(idx)}]{_dim_name(info, feature, int(idx))}")
    if not labels:
        return None
    return make_issue(
        "l2.constant_channel",
        "warn",
        ep.episode_index,
        0,
        max(0, ep.length - 1),
        1,
        "episode-specific constant channel(s): " + ", ".join(labels[:8]),
    )


def step_baseline(
    data: dict[int, EpisodeArrays],
    feature: str,
    lo: int,
    hi: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray] | None:
    diffs = []
    for arrays in data.values():
        matrix = getattr(arrays, feature)
        if matrix.shape[0] >= 2:
            diff = np.diff(matrix[:, lo : hi + 1], axis=0)
            diff = np.where(np.isfinite(diff), diff, np.nan)
            diffs.append(diff)
    if not diffs:
        return None
    pooled = np.concatenate(diffs, axis=0)
    median = np.nanmedian(pooled, axis=0)
    mad = np.nanmedian(np.abs(pooled - median), axis=0)
    scale = np.maximum(1.4826 * mad, cfg.step_scale_floor)
    return median, scale


def _feature_step_hits(
    matrix: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    cfg: Config,
    info: dict,
    feature: str,
    lo: int,
) -> tuple[np.ndarray, str] | None:
    if matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        return None
    diffs = np.diff(matrix[:, lo : lo + len(median)], axis=0)
    if diffs.size == 0:
        return None
    z = np.abs(diffs - median) / scale
    mask = (z > cfg.step_z) & (np.abs(diffs) > cfg.step_abs_floor)
    transitions = np.where(mask.any(axis=1))[0]
    if transitions.size == 0:
        return None
    masked_abs = np.where(mask, np.abs(diffs), -np.inf)
    flat = int(np.argmax(masked_abs))
    t, j = np.unravel_index(flat, mask.shape)
    detail = (
        f"{feature}: {transitions.size} step transition(s), max |Δ|={abs(diffs[t, j]):.4f} "
        f"at frame {t}->{t + 1} ({_dim_name(info, feature, lo + int(j))})"
    )
    return transitions, detail


def check_joint_steps(
    ep: EpisodeMeta,
    arrays: EpisodeArrays,
    info: dict,
    action_baseline: tuple[np.ndarray, np.ndarray] | None,
    state_baseline: tuple[np.ndarray, np.ndarray] | None,
    cfg: Config,
) -> Issue | None:
    lo, hi = cfg.joint_dims
    transitions: list[np.ndarray] = []
    details: list[str] = []

    if action_baseline is not None:
        hit = _feature_step_hits(arrays.action, *action_baseline, cfg, info, "action", lo)
        if hit is not None:
            transitions.append(hit[0])
            details.append(hit[1])
    if state_baseline is not None:
        hit = _feature_step_hits(arrays.state, *state_baseline, cfg, info, "state", lo)
        if hit is not None:
            transitions.append(hit[0])
            details.append(hit[1])

    if not details:
        return None
    all_transitions = np.unique(np.concatenate(transitions)) if transitions else np.array([], dtype=int)
    event_count = int(sum(t.size for t in transitions))
    frames = transition_frames(all_transitions.tolist(), ep.length)
    span = frame_span(frames, ep.length)
    if span is None:
        return None
    start, end, _ = span
    severity = "fail" if event_count >= cfg.step_fail_count else "warn"
    return make_issue(
        "l2.joint_step",
        severity,
        ep.episode_index,
        start,
        end,
        event_count,
        "; ".join(details),
    )


def check_action_state_residual(
    ep: EpisodeMeta,
    arrays: EpisodeArrays,
    cfg: Config,
) -> Issue | None:
    dim = min(18, arrays.action.shape[1], arrays.state.shape[1])
    if dim <= 0 or arrays.num_rows == 0:
        return None
    residual = np.abs(arrays.action[:, :dim] - arrays.state[:, :dim])
    if not np.isfinite(residual).all():
        return None
    p99 = float(np.nanpercentile(residual, 99))
    maximum = float(np.nanmax(residual))
    if p99 <= cfg.action_state_p99_warn and maximum <= cfg.action_state_max_warn:
        return None
    frame_max = np.nanmax(residual, axis=1)
    frames = np.where(frame_max > cfg.action_state_frame_tol)[0]
    if frames.size == 0:
        frames = np.array([int(np.nanargmax(frame_max))], dtype=int)
    flat = int(np.nanargmax(residual))
    worst_frame, worst_dim = np.unravel_index(flat, residual.shape)
    start, end, _ = frame_span(frames, ep.length) or (int(worst_frame), int(worst_frame), 1)
    return make_issue(
        "l2.action_state_residual",
        "warn",
        ep.episode_index,
        start,
        end,
        int(frames.size),
        f"same-frame |action-state| p99={p99:.4f}, max={maximum:.4f} at frame {worst_frame} "
        f"(dim {worst_dim}); frames over {cfg.action_state_frame_tol}: "
        f"{examples_text(frames, cfg.max_examples_in_detail)}",
    )
