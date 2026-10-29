from __future__ import annotations

import numpy as np
import pytest

from dclean.dataset import parse_episode_spec, resolve_data_path, resolve_video_path


def test_parse_episode_spec_ranges_and_dedup():
    assert parse_episode_spec("0-3,2,5", 10) == [0, 1, 2, 3, 5]
    assert parse_episode_spec("all", 4) == [0, 1, 2, 3]
    assert parse_episode_spec(None, 3) == [0, 1, 2]
    assert parse_episode_spec(" 7 ", 10) == [7]


@pytest.mark.parametrize("spec", ["", "3-1", "-1", "10", "0-", "a", "1,,2"])
def test_parse_episode_spec_invalid(spec):
    with pytest.raises(ValueError):
        parse_episode_spec(spec, 10)


def test_metadata_and_paths(loaded, synthetic_root):
    info, tasks, episodes, arrays = loaded
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 10
    assert info["total_frames"] == 300
    assert tasks == {0: "Part Sorting"}
    assert [ep.episode_index for ep in episodes] == list(range(10))
    assert all(ep.length == 30 for ep in episodes)

    ep0 = episodes[0]
    assert resolve_data_path(synthetic_root, info, ep0).name == "file-000.parquet"
    assert resolve_video_path(synthetic_root, info, ep0, "observation.images.head_left").name == "file-000.mp4"

    # wrist_right is split at episode 5; per-camera offsets are independent.
    ep5 = episodes[5]
    wr = ep5.videos["observation.images.wrist_right"]
    assert wr.file_index == 1
    assert wr.from_timestamp == pytest.approx(0.0)
    hl = ep5.videos["observation.images.head_left"]
    assert hl.file_index == 0
    assert hl.from_timestamp == pytest.approx(5.0)

    # ep9 intentionally references a missing video file.
    assert episodes[9].videos["observation.images.wrist_right"].file_index == 9


def test_arrays_truncate_state_payload(loaded):
    info, tasks, episodes, arrays = loaded
    ep0 = arrays[0]
    assert ep0.action.shape == (30, 20)
    assert ep0.state.shape == (30, 20)  # state[20:48] is discarded
    assert np.isfinite(ep0.state).all()  # fixture puts NaN at state[20] only

    assert ep0.timestamp[0] == 0.0
    assert ep0.timestamp[-1] == pytest.approx(29.0 / 30.0, abs=1e-6)
    assert np.array_equal(ep0.frame_index, np.arange(30))
    assert np.all(ep0.episode_index == 0)


def test_multi_episode_file_extraction(loaded):
    info, tasks, episodes, arrays = loaded
    ep5 = arrays[5]
    assert ep5.action.shape == (30, 20)
    assert np.all(ep5.episode_index == 5)
    assert ep5.index[0] == 150
    assert ep5.index[-1] == 179
    assert ep5.frame_index[0] == 0
    assert ep5.frame_index[-1] == 29
