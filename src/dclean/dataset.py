"""LeRobot Dataset v3.0 metadata and data access.

The central v3.0 rule used here: episode boundaries live in
``meta/episodes/chunk-*/file-*.parquet``; data/video file names do not identify
episodes. Every lookup goes through the metadata, and video paths are resolved
independently per camera.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATA_COLUMNS = [
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]

# Only the first 20 state dimensions are checked. state[20:48] is object pose
# payload and is intentionally discarded at read time.
CHECKED_STATE_DIMS = 20


@dataclass(frozen=True)
class VideoSegment:
    camera: str
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    videos: dict[str, VideoSegment]


@dataclass(frozen=True)
class EpisodeArrays:
    action: np.ndarray         # (T, 20) float64
    state: np.ndarray          # (T, 20) float64, only the checked prefix
    timestamp: np.ndarray      # (T,) float64
    frame_index: np.ndarray    # (T,) int64
    episode_index: np.ndarray  # (T,) int64
    index: np.ndarray          # (T,) int64
    task_index: np.ndarray     # (T,) int64

    @property
    def num_rows(self) -> int:
        return int(self.action.shape[0])


def load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3.0"):
        raise ValueError(f"expected LeRobot v3.0 dataset, got {version!r}")
    for key in ("fps", "total_episodes", "total_frames", "total_tasks"):
        if key not in info:
            raise ValueError(f"meta/info.json missing required key {key!r}")
    return info


def video_cameras(info: dict) -> list[str]:
    return [
        key
        for key, spec in info.get("features", {}).items()
        if spec.get("dtype") == "video"
    ]


def load_tasks(root: Path) -> dict[int, str]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    table = pq.read_table(path)
    cols = set(table.column_names)
    if "task_index" not in cols:
        raise ValueError("meta/tasks.parquet has no 'task_index' column")
    if "task" in cols:
        task_col = "task"
    elif "__index_level_0__" in cols:
        task_col = "__index_level_0__"
    else:
        raise ValueError("meta/tasks.parquet has no task text column")
    tasks: dict[int, str] = {}
    for row in table.select(["task_index", task_col]).to_pylist():
        tasks[int(row["task_index"])] = str(row[task_col])
    return tasks


def load_episodes(root: Path, info: dict) -> list[EpisodeMeta]:
    paths = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    cameras = video_cameras(info)

    tables = []
    for path in paths:
        pf = pq.ParquetFile(path)
        columns = [name for name in pf.schema_arrow.names if not name.startswith("stats/")]
        tables.append(pf.read(columns=columns))
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    required = {
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"episode metadata missing columns: {sorted(missing)}")

    episodes: list[EpisodeMeta] = []
    for row in table.to_pylist():
        videos: dict[str, VideoSegment] = {}
        for camera in cameras:
            prefix = f"videos/{camera}/"
            try:
                videos[camera] = VideoSegment(
                    camera=camera,
                    chunk_index=int(row[prefix + "chunk_index"]),
                    file_index=int(row[prefix + "file_index"]),
                    from_timestamp=float(row[prefix + "from_timestamp"]),
                    to_timestamp=float(row[prefix + "to_timestamp"]),
                )
            except KeyError as exc:
                raise ValueError(
                    f"episode metadata missing video columns for camera {camera!r}"
                ) from exc
        episodes.append(
            EpisodeMeta(
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                tasks=tuple(str(t) for t in (row.get("tasks") or [])),
                data_chunk_index=int(row["data/chunk_index"]),
                data_file_index=int(row["data/file_index"]),
                dataset_from_index=int(row["dataset_from_index"]),
                dataset_to_index=int(row["dataset_to_index"]),
                videos=videos,
            )
        )

    episodes.sort(key=lambda ep: ep.episode_index)
    return episodes


def resolve_data_path(root: Path, info: dict, ep: EpisodeMeta) -> Path:
    rel = info["data_path"].format(
        chunk_index=ep.data_chunk_index,
        file_index=ep.data_file_index,
    )
    return root / rel


def resolve_video_path(root: Path, info: dict, ep: EpisodeMeta, camera: str) -> Path:
    seg = ep.videos[camera]
    rel = info["video_path"].format(
        video_key=camera,
        chunk_index=seg.chunk_index,
        file_index=seg.file_index,
    )
    return root / rel


def parse_episode_spec(spec: str | None, total: int) -> list[int]:
    """Parse ``0-9,15,20-22`` (or ``all``/None) into a sorted unique list."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(total))
    if total <= 0:
        raise ValueError("dataset has no episodes")

    selected: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            raise ValueError(f"invalid episode spec: {spec!r}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"invalid episode range: {token!r}")
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"invalid episode range (start > end): {token!r}")
            values = range(start, end + 1)
        else:
            values = [int(token)]
        for value in values:
            if value < 0 or value >= total:
                raise ValueError(f"episode {value} out of range [0, {total - 1}]")
            selected.add(value)
    if not selected:
        raise ValueError("no episodes selected")
    return sorted(selected)


def read_data_table(path: Path) -> pa.Table:
    if not path.is_file():
        raise FileNotFoundError(f"missing data file {path}")
    table = pq.read_table(path)
    missing = set(DEFAULT_DATA_COLUMNS) - set(table.column_names)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return table.select(DEFAULT_DATA_COLUMNS)


def _fixed_list_to_numpy(column: pa.ChunkedArray, dim: int) -> np.ndarray:
    arr = column.combine_chunks()
    flat = arr.flatten()
    values = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float64)
    return values.reshape(len(arr), dim)


def extract_episode_arrays(
    table: pa.Table,
    ep: EpisodeMeta,
    *,
    action_dim: int = 20,
    state_dim: int = 48,
) -> EpisodeArrays:
    """Extract one episode's rows from a data table.

    Primary lookup is the ``episode_index`` column, which supports multiple
    episodes per parquet. If that is empty (corrupt metadata), fall back to the
    global ``index`` range. Row-count mismatches are reported by L0; this
    function never silently renumbers rows.
    """
    ep_col = np.asarray(table.column("episode_index").to_numpy(zero_copy_only=False))
    mask = ep_col == ep.episode_index
    sub = table.filter(pa.array(mask))
    if sub.num_rows == 0:
        idx = np.asarray(table.column("index").to_numpy(zero_copy_only=False))
        mask = (idx >= ep.dataset_from_index) & (idx < ep.dataset_to_index)
        sub = table.filter(pa.array(mask))

    action = _fixed_list_to_numpy(sub.column("action"), action_dim)
    state_all = _fixed_list_to_numpy(sub.column("observation.state"), state_dim)
    if state_all.shape[1] < CHECKED_STATE_DIMS:
        raise ValueError(
            f"observation.state has {state_all.shape[1]} dims, expected >= {CHECKED_STATE_DIMS}"
        )
    state = np.ascontiguousarray(state_all[:, :CHECKED_STATE_DIMS])

    return EpisodeArrays(
        action=action,
        state=state,
        timestamp=np.asarray(
            sub.column("timestamp").to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        frame_index=np.asarray(
            sub.column("frame_index").to_numpy(zero_copy_only=False), dtype=np.int64
        ),
        episode_index=np.asarray(
            sub.column("episode_index").to_numpy(zero_copy_only=False), dtype=np.int64
        ),
        index=np.asarray(sub.column("index").to_numpy(zero_copy_only=False), dtype=np.int64),
        task_index=np.asarray(
            sub.column("task_index").to_numpy(zero_copy_only=False), dtype=np.int64
        ),
    )


def group_episodes_by_data_file(
    root: Path, info: dict, episodes: list[EpisodeMeta]
) -> dict[Path, list[EpisodeMeta]]:
    groups: dict[Path, list[EpisodeMeta]] = {}
    for ep in episodes:
        groups.setdefault(resolve_data_path(root, info, ep), []).append(ep)
    return groups
