"""Statistics used by the cleaning stage.

Low-dimensional features are recomputed exactly from the retained rows.
Image/video statistics are not decoded again: their per-episode values are
copied from ``meta/episodes`` and aggregated with the same weighted formulas
LeRobot uses in ``aggregate_stats``.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILES = (1.0, 10.0, 50.0, 90.0, 99.0)
QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")


def column_to_numpy(column: pa.ChunkedArray, dim: int) -> np.ndarray:
    """Convert a table column to ``(n, dim)`` float64.

    Supports scalar columns (``dim == 1``) and Arrow fixed-size-list columns
    (action/state). Raises on shape/type mismatches instead of guessing.
    """
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_fixed_size_list(arr.type):
        if int(arr.type.list_size) != dim:
            raise ValueError(
                f"column has fixed list size {arr.type.list_size}, expected {dim}"
            )
        flat = arr.flatten()
        values = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values.reshape(len(arr), dim)
    values = np.asarray(column.to_numpy(zero_copy_only=False))
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] != dim:
        raise ValueError(f"column has {values.shape[1]} values per row, expected {dim}")
    return values.astype(np.float64, copy=False)


def compute_stats(values: np.ndarray, *, require_finite: bool = True) -> dict[str, np.ndarray]:
    """Exact min/max/mean/std/quantiles for ``(n, dim)`` values.

    ``count`` is the number of rows (frames), matching LeRobot's per-episode
    and dataset-level convention. ``std`` uses ``ddof=0`` like ``np.std``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("cannot compute statistics for zero rows")
    if require_finite and not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"cannot compute statistics: {bad} non-finite value(s)")

    stats: dict[str, np.ndarray] = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    for key, quantile in zip(QUANTILE_KEYS, QUANTILES):
        stats[key] = np.percentile(arr, quantile, axis=0)
    return stats


def stats_to_json(stats: dict[str, np.ndarray]) -> dict[str, list]:
    """Convert numpy statistics to JSON-friendly lists without flattening.

    Low-dimensional stats have shape ``(dim,)`` and become flat lists. Image
    stats have shape ``(3, 1, 1)`` and keep that nesting, matching LeRobot's
    ``stats.json`` layout.
    """
    payload: dict[str, list] = {}
    for key in STAT_KEYS:
        if key not in stats:
            continue
        values = np.asarray(stats[key])
        if key == "count":
            payload[key] = np.asarray(values, dtype=np.int64).tolist()
        else:
            payload[key] = np.asarray(values, dtype=np.float64).tolist()
    return payload


def image_stats_from_row(row: dict, camera: str) -> dict[str, np.ndarray] | None:
    """Extract one episode's image stats for ``camera`` from metadata row.

    Returns ``None`` when the source metadata does not carry the complete set
    of image statistics. This keeps old datasets (stats not stored) usable.
    """
    prefix = f"stats/{camera}/"
    if any(prefix + key not in row or row[prefix + key] is None for key in STAT_KEYS):
        return None
    values: dict[str, np.ndarray] = {}
    for key in STAT_KEYS:
        raw = row[prefix + key]
        if key == "count":
            values[key] = np.asarray(raw, dtype=np.int64).reshape(-1)
        else:
            values[key] = np.asarray(raw, dtype=np.float64)
    return values


def aggregate_image_stats(stats_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Aggregate per-episode image stats with LeRobot's weighted formulas.

    - min/max: element-wise min/max;
    - mean/quantiles: weighted by pixel count;
    - std: combined variance;
    - count: sum.

    Quantiles cannot be reconstructed exactly from per-episode quantiles, so
    this deliberately mirrors LeRobot's approximation rather than pretending
    to be exact.
    """
    if not stats_list:
        raise ValueError("cannot aggregate an empty stats list")

    counts = np.asarray(
        [float(np.asarray(item["count"]).reshape(-1)[0]) for item in stats_list],
        dtype=np.float64,
    )
    if not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("image stats contain invalid counts")
    total_count = float(counts.sum())

    means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in stats_list])
    variances = np.stack(
        [np.asarray(item["std"], dtype=np.float64) ** 2 for item in stats_list]
    )
    broadcast_shape = (len(counts),) + (1,) * (means.ndim - 1)
    weights = counts.reshape(broadcast_shape)

    total_mean = (means * weights).sum(axis=0) / total_count
    delta = means - total_mean
    total_variance = ((variances + delta**2) * weights).sum(axis=0) / total_count

    aggregate: dict[str, np.ndarray] = {
        "min": np.min(
            np.stack([np.asarray(item["min"], dtype=np.float64) for item in stats_list]),
            axis=0,
        ),
        "max": np.max(
            np.stack([np.asarray(item["max"], dtype=np.float64) for item in stats_list]),
            axis=0,
        ),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": np.array([int(total_count)], dtype=np.int64),
    }
    for key in QUANTILE_KEYS:
        if all(key in item for item in stats_list):
            values = np.stack(
                [np.asarray(item[key], dtype=np.float64) for item in stats_list]
            )
            aggregate[key] = (values * weights).sum(axis=0) / total_count
    return aggregate
