from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dclean.config import load_config
from dclean.dataset import (
    extract_episode_arrays,
    load_episodes,
    load_info,
    load_tasks,
    read_data_table,
    resolve_data_path,
)
from tests.fixtures import make_synthetic_dataset, snapshot_tree


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to build the synthetic video fixtures")
    root = tmp_path_factory.mktemp("synthetic_v30") / "dataset"
    return make_synthetic_dataset(root)


@pytest.fixture(scope="session")
def cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "part_sorting.toml")


@pytest.fixture(scope="session")
def synthetic_snapshot(synthetic_root: Path) -> dict[str, tuple[int, int]]:
    return snapshot_tree(synthetic_root)


@pytest.fixture(scope="session")
def loaded(synthetic_root: Path):
    info = load_info(synthetic_root)
    tasks = load_tasks(synthetic_root)
    episodes = load_episodes(synthetic_root, info)
    table = read_data_table(resolve_data_path(synthetic_root, info, episodes[0]))
    arrays = {ep.episode_index: extract_episode_arrays(table, ep) for ep in episodes}
    return info, tasks, episodes, arrays
