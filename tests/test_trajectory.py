from __future__ import annotations

import numpy as np

from dclean.dataset import EpisodeArrays
from dclean.trajectory import calibrate_trajectory, check_trajectory_deviation


INFO = {"features": {"action": {"names": [f"j{i}" for i in range(20)]}}}


def _make_arrays(action: np.ndarray) -> EpisodeArrays:
    action = np.asarray(action, dtype=np.float64)
    n = action.shape[0]
    return EpisodeArrays(
        action=action,
        state=action.copy(),
        timestamp=np.arange(n, dtype=np.float64) / 30.0,
        frame_index=np.arange(n, dtype=np.int64),
        episode_index=np.zeros(n, dtype=np.int64),
        index=np.arange(n, dtype=np.int64),
        task_index=np.zeros(n, dtype=np.int64),
    )


def _dataset(n_episodes: int = 12, length: int = 120) -> dict[int, EpisodeArrays]:
    time = np.arange(length, dtype=np.float64)
    base = 0.3 * np.sin(2 * np.pi * time / 60.0)[:, None] * np.ones((1, 20))
    rng = np.random.default_rng(3)
    data = {
        ep: _make_arrays(base + 1e-4 * rng.normal(size=base.shape))
        for ep in range(n_episodes)
    }
    # Sustained, smoothly ramped 1.2 rad offset on j3 for frames 40..109.
    # The 20-frame ramps keep per-frame steps at 0.06 (< the step threshold).
    offset = np.zeros(length)
    offset[40:60] = np.linspace(0.0, 1.2, 20)
    offset[60:90] = 1.2
    offset[90:110] = np.linspace(1.2, 0.0, 20)
    data[5] = _make_arrays(base + 1e-4 * rng.normal(size=base.shape))
    data[5].action[:, 3] += offset
    return data


def test_trajectory_deviation_finds_sustained_offset(cfg):
    data = _dataset()
    result = check_trajectory_deviation(data, INFO, cfg)
    assert result.dataset_issue is not None
    assert result.dataset_issue.issue_id == "l2.trajectory_baseline"
    assert "12 episodes" in result.dataset_issue.detail
    assert len(result.metrics) == 12

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.episode_index == 5
    assert issue.issue_id == "l2.trajectory_deviation"
    assert issue.severity == "fail"
    assert 40 <= issue.frame_start <= 60
    assert issue.frame_end >= 90
    assert "j3" in issue.detail
    assert issue.event_count >= 30

    anomaly = next(m for m in result.metrics if m.episode_index == 5)
    assert anomaly.computed and anomaly.flagged
    assert anomaly.segment_start == issue.frame_start
    assert anomaly.segment_end == issue.frame_end
    assert anomaly.max_smoothed_score is not None and anomaly.max_smoothed_score > 20
    clean = next(m for m in result.metrics if m.episode_index == 0)
    assert clean.computed and not clean.flagged

    calibration = calibrate_trajectory(result.metrics, cfg.trajectory_z_warn)
    assert calibration["computed_episodes"] == 12
    assert calibration["flagged_episodes"] == [5]
    assert calibration["suggested_threshold"] is not None
    assert calibration["largest_gap"] is not None
    assert calibration["largest_gap"]["high"] > calibration["largest_gap"]["low"]


def test_trajectory_requires_enough_episodes(cfg):
    data = {ep: arrays for ep, arrays in list(_dataset().items())[:5]}
    result = check_trajectory_deviation(data, INFO, cfg)
    assert result.issues == []
    assert result.dataset_issue is not None
    assert result.dataset_issue.issue_id == "l2.trajectory_not_computed"
    assert result.dataset_issue.severity == "info"
    assert len(result.metrics) == 5
    assert all(not metric.computed for metric in result.metrics)
    assert all("need at least" in metric.note for metric in result.metrics)
    calibration = calibrate_trajectory(result.metrics, cfg.trajectory_z_warn)
    assert calibration["computed_episodes"] == 0


def test_trajectory_requires_equal_lengths(cfg):
    data = _dataset()
    data[11] = _make_arrays(np.zeros((121, 20)))
    result = check_trajectory_deviation(data, INFO, cfg)
    assert result.issues == []
    assert result.dataset_issue is not None
    assert result.dataset_issue.issue_id == "l2.trajectory_not_computed"
    assert "variable episode lengths" in result.dataset_issue.detail
    assert len(result.metrics) == 12
    assert all(not metric.computed for metric in result.metrics)
