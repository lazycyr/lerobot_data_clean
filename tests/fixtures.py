"""Build a small synthetic LeRobot v3.0 dataset for tests.

Schema mirrors Part_Sorting_300 (4 cameras, action[20], state[48], 30 fps) but
uses 10 episodes x 30 frames so tests stay fast. The 10 episodes intentionally
cover common data problems.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = [
    "observation.images.head_left",
    "observation.images.head_right",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
]
TASK = "Part Sorting"


def _fixed_list(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    flat = pa.array(values.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _make_video(path: Path, frames: int, *, fps: int, codec: str = "libx264") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s=64x64:r={fps}",
        "-frames:v",
        str(frames),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if codec == "libx264":
        cmd += ["-preset", "ultrafast"]
    elif codec == "libsvtav1":
        cmd += ["-preset", "13"]
    elif codec == "libaom-av1":
        cmd += ["-cpu-used", "8"]
    cmd.append(str(path))
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {completed.stderr.strip()}")


def _features(info_cameras: list[str]) -> dict:
    joint_names = [f"joint_{i}.pos" for i in range(20)]
    object_names = [
        f"object_{obj}_{axis}"
        for obj in range(1, 5)
        for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")
    ]
    features: dict = {
        "action": {"dtype": "float32", "shape": [20], "names": joint_names},
        "observation.state": {
            "dtype": "float32",
            "shape": [48],
            "names": joint_names + object_names,
        },
    }
    for camera in info_cameras:
        features[camera] = {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 64,
                "video.width": 64,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
        }
    )
    return features


def make_synthetic_dataset(
    root: Path,
    *,
    n_episodes: int = 10,
    length: int = 30,
    fps: int = 30,
    seed: int = 7,
    inject_problems: bool = True,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for camera in CAMERAS:
        (root / "videos" / camera / "chunk-000").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    total = n_episodes * length
    action = np.zeros((total, 20), dtype=np.float32)
    state = np.zeros((total, 48), dtype=np.float32)
    timestamps = np.zeros(total, dtype=np.float32)
    frame_index = np.zeros(total, dtype=np.int64)
    episode_index = np.zeros(total, dtype=np.int64)
    global_index = np.arange(total, dtype=np.int64)
    task_index = np.zeros(total, dtype=np.int64)

    for ep in range(n_episodes):
        sl = slice(ep * length, (ep + 1) * length)
        t = np.arange(length, dtype=np.float64) / fps
        phase = rng.uniform(0, 2 * np.pi, size=20)
        freq = rng.uniform(0.2, 1.0, size=20)
        amp = rng.uniform(0.2, 1.0, size=20)
        base = amp[None, :] * np.sin(2 * np.pi * freq[None, :] * t[:, None] + phase[None, :])
        base += 0.005 * rng.normal(size=(length, 20))

        # Discrete grippers: 18 is constant 1; 19 toggles -1 -> +1.
        base[:, 18] = 1.0
        base[:, 19] = np.where(t < (length / fps) / 2.0, -1.0, 1.0)

        act = base.copy()
        st = base + 0.0003 * rng.normal(size=(length, 20))
        st[:, 18] = act[:, 18]
        st[:, 19] = act[:, 19]
        ts = np.arange(length, dtype=np.float64) / fps

        # --- injected problems -------------------------------------------
        if inject_problems:
            if ep == 2:
                act[10:13, 3] = np.nan          # NaN in action
                st[20, 7] = np.inf              # Inf in checked state dim
            if ep == 3:
                # non-monotonic transition at 14->15, then keep dt normal
                ts[15] = ts[14] - 0.001
                ts[16:] -= (1.0 / fps + 0.001)
                # duplicate timestamp at 21->22, then keep dt normal
                ts[22] = ts[21]
                ts[23:] -= 1.0 / fps
            if ep == 4:
                ts[8:] += 1.0 / fps             # missing frame / dt gap
            if ep == 5:
                act[12:, 3] += 0.5              # joint step in action
                st[12:, 3] += 0.5               # keep action/state aligned
            if ep == 6:
                act[:, 9] = act[0, 9]           # episode-specific constant channel
                st[:, 9] = st[0, 9]
            if ep == 7:
                offset = np.zeros(length)
                offset[5:10] = np.array([0.03, 0.06, 0.09, 0.12, 0.12])
                act[:, 5] += offset             # smooth action-state residual

        action[sl] = act.astype(np.float32)
        state[sl, :20] = st.astype(np.float32)
        state[sl, 20:] = 0.0               # object pose payload, ignored by check
        if inject_problems and ep == 0:
            state[sl, 20] = np.nan         # must not affect ep0: payload is ignored
        timestamps[sl] = ts.astype(np.float32)
        frame_index[sl] = np.arange(length)
        episode_index[sl] = ep

    if inject_problems:
        # ep9: corrupt global index (all rows +1).
        global_index[9 * length : 10 * length] += 1

    data = pa.table(
        {
            "action": _fixed_list(action, 20),
            "observation.state": _fixed_list(state, 48),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(data, data_path)
    if inject_problems:
        # Orphan data file: exists but no episode references it.
        shutil.copy2(data_path, root / "data" / "chunk-000" / "file-001.parquet")

    # --- episode metadata -------------------------------------------------
    rows: list[dict] = []
    for ep in range(n_episodes):
        start_index = ep * length
        end_index = start_index + length
        row: dict = {
            "episode_index": ep,
            "tasks": [TASK],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start_index,
            "dataset_to_index": end_index,
            "stats/action/count": [length],
            "stats/observation.state/count": [length],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera in CAMERAS:
            if camera.endswith("wrist_right"):
                file_index = 0 if ep < 5 else 1
                start_s = (ep if ep < 5 else ep - 5) * length / fps
            else:
                file_index = 0
                start_s = ep * length / fps
            end_s = start_s + length / fps
            if inject_problems and ep == 8 and camera.endswith("head_left"):
                end_s += 2.0 / fps           # segment duration mismatch
            if inject_problems and ep == 9 and camera.endswith("wrist_right"):
                file_index = 9               # missing referenced video
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = file_index
            row[f"videos/{camera}/from_timestamp"] = float(start_s)
            row[f"videos/{camera}/to_timestamp"] = float(end_s)

            # Per-episode image statistics. Values vary slightly by episode so
            # the global weighted aggregation is observable in tests.
            center = 0.4 + 0.01 * ep
            row[f"stats/{camera}/min"] = [[[0.0]], [[0.0]], [[0.0]]]
            row[f"stats/{camera}/max"] = [[[1.0]], [[1.0]], [[1.0]]]
            row[f"stats/{camera}/mean"] = [[[center]], [[center]], [[center]]]
            row[f"stats/{camera}/std"] = [[[0.1]], [[0.1]], [[0.1]]]
            row[f"stats/{camera}/count"] = [length * 64 * 64]
            row[f"stats/{camera}/q01"] = [[[0.05]], [[0.05]], [[0.05]]]
            row[f"stats/{camera}/q10"] = [[[0.15]], [[0.15]], [[0.15]]]
            row[f"stats/{camera}/q50"] = [[[0.5]], [[0.5]], [[0.5]]]
            row[f"stats/{camera}/q90"] = [[[0.85]], [[0.85]], [[0.85]]]
            row[f"stats/{camera}/q99"] = [[[0.95]], [[0.95]], [[0.95]]]
        rows.append(row)

    pq.write_table(pa.Table.from_pylist(rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([TASK], type=pa.string()),
            }
        ),
        root / "meta" / "tasks.parquet",
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic_walker",
        "total_episodes": n_episodes,
        "total_frames": total,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": _features(CAMERAS),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "action": {"count": [total]},
                "observation.state": {"count": [total]},
                "timestamp": {"count": [total]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --- videos ----------------------------------------------------------
    # One file per camera for the first three cameras; wrist_right is split at
    # episode 5 to exercise independent per-camera file/timestamp offsets.
    _make_video(root / "videos" / CAMERAS[0] / "chunk-000" / "file-000.mp4", total, fps=fps)
    _make_video(root / "videos" / CAMERAS[1] / "chunk-000" / "file-000.mp4", total, fps=fps)
    _make_video(root / "videos" / CAMERAS[2] / "chunk-000" / "file-000.mp4", total, fps=fps)
    # wrist_right file-000 covers episodes 0-4 (5 x 30 frames); file-001 covers
    # episodes 5-8 (4 x 30 frames). Episode 9's metadata points at a missing
    # file-009, so it does not contribute to file-001's expected frame count.
    _make_video(
        root / "videos" / CAMERAS[3] / "chunk-000" / "file-000.mp4",
        5 * length,
        fps=fps,
    )
    _make_video(
        root / "videos" / CAMERAS[3] / "chunk-000" / "file-001.mp4",
        (5 if not inject_problems else 4) * length,
        fps=fps,
        codec="libsvtav1",  # codec mismatch vs info.json's h264
    )

    return root


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return result
