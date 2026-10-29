from __future__ import annotations

import numpy as np
import pytest

from dclean.checks import (
    check_action_state_residual,
    check_constant_channels,
    check_episode_structure,
    check_finite,
    check_joint_steps,
    check_timestamps,
    global_constant_issue,
    global_dim_stats,
    step_baseline,
)
from dclean.dataset import EpisodeArrays
from dclean.scoring import make_issue, merge_issues, suppress_covered_step_issues


def _make_arrays(action: np.ndarray, state: np.ndarray | None = None) -> EpisodeArrays:
    action = np.asarray(action, dtype=np.float64)
    state = action.copy() if state is None else np.asarray(state, dtype=np.float64)
    n = action.shape[0]
    return EpisodeArrays(
        action=action,
        state=state,
        timestamp=np.arange(n, dtype=np.float64) / 30.0,
        frame_index=np.arange(n, dtype=np.int64),
        episode_index=np.zeros(n, dtype=np.int64),
        index=np.arange(n, dtype=np.int64),
        task_index=np.zeros(n, dtype=np.int64),
    )


def test_episode_structure_clean_and_bad_index(loaded):
    info, tasks, episodes, arrays = loaded
    assert check_episode_structure(episodes[0], arrays[0], tasks) == []
    issues = check_episode_structure(episodes[9], arrays[9], tasks)
    global_index = [i for i in issues if i.issue_id == "l0.global_index"]
    assert len(global_index) == 1
    assert global_index[0].severity == "fail"
    assert global_index[0].event_count == 30
    assert "270..299" in global_index[0].detail
    assert "271..300" in global_index[0].detail


def test_timestamps(loaded, cfg):
    info, tasks, episodes, arrays = loaded
    assert check_timestamps(episodes[0], arrays[0], cfg) == []

    ep3 = {i.issue_id: i for i in check_timestamps(episodes[3], arrays[3], cfg)}
    assert "l1.timestamp_nonmonotonic" in ep3
    assert "l1.timestamp_duplicate" in ep3
    assert ep3["l1.timestamp_nonmonotonic"].frame_start == 14
    assert ep3["l1.timestamp_duplicate"].frame_start == 21

    ep4 = {i.issue_id: i for i in check_timestamps(episodes[4], arrays[4], cfg)}
    assert "l1.timestamp_gap" in ep4
    assert ep4["l1.timestamp_gap"].frame_start == 7


def test_finite_ignores_object_payload(loaded):
    info, tasks, episodes, arrays = loaded
    issue = check_finite(episodes[2], arrays[2])
    assert issue is not None
    assert issue.issue_id == "l2.nan_inf"
    assert issue.severity == "fail"
    assert issue.frame_start == 10
    assert issue.frame_end == 20
    assert issue.event_count == 4
    # ep0 has NaN at full-state dim 20; it must be invisible after truncation.
    assert arrays[0].state.shape[1] == 20
    assert check_finite(episodes[0], arrays[0]) is None


def test_global_constant_and_episode_constant(loaded, cfg):
    info, tasks, episodes, arrays = loaded
    action_std, action_frac, action_global = global_dim_stats(arrays, "action", cfg)
    state_std, state_frac, state_global = global_dim_stats(arrays, "state", cfg)
    assert action_global[18]
    assert state_global[18]
    issue = global_constant_issue(info, "action", action_global)
    assert issue is not None and issue.severity == "info"
    assert "joint_18.pos" in issue.detail

    issue = check_constant_channels(
        episodes[6],
        arrays[6],
        info,
        action_std,
        action_global,
        state_std,
        state_global,
        cfg,
    )
    assert issue is not None
    assert issue.issue_id == "l2.constant_channel"
    assert issue.severity == "warn"
    assert "joint_9.pos" in issue.detail
    assert issue.frame_start == 0 and issue.frame_end == 29

    assert (
        check_constant_channels(
            episodes[0],
            arrays[0],
            info,
            action_std,
            action_global,
            state_std,
            state_global,
            cfg,
        )
        is None
    )


def test_joint_step_detection(loaded, cfg):
    info, tasks, episodes, arrays = loaded
    lo, hi = cfg.joint_dims
    action_base = step_baseline(arrays, "action", lo, hi, cfg)
    state_base = step_baseline(arrays, "state", lo, hi, cfg)
    assert action_base is not None and state_base is not None

    issue = check_joint_steps(episodes[5], arrays[5], info, action_base, state_base, cfg)
    assert issue is not None
    assert issue.issue_id == "l2.joint_step"
    assert issue.severity == "warn"
    assert issue.frame_start <= 11 <= issue.frame_end
    assert issue.event_count >= 1

    assert check_joint_steps(episodes[0], arrays[0], info, action_base, state_base, cfg) is None


def test_step_scale_floor_and_discrete_dims(cfg):
    # Constant data: MAD is zero, so the scale floor must prevent huge z-scores.
    action = np.zeros((12, 20), dtype=np.float64)
    base = _make_arrays(action)
    median, scale = step_baseline({0: base}, "action", 0, 17, cfg)
    assert np.all(scale >= cfg.step_scale_floor - 1e-12)

    # A huge jump on gripper dim 18 must not be reported as a joint step.
    jumped = action.copy()
    jumped[6:, 18] = 2.0
    arrays = {0: base, 1: _make_arrays(jumped)}
    median, scale = step_baseline(arrays, "action", 0, 17, cfg)
    issue = check_joint_steps(
        type("Ep", (), {"episode_index": 1, "length": 12})(),
        arrays[1],
        {"features": {"action": {"names": [f"j{i}" for i in range(20)]}}},
        (median, scale),
        (median, scale),
        cfg,
    )
    assert issue is None


def test_action_state_residual(loaded, cfg):
    info, tasks, episodes, arrays = loaded
    issue = check_action_state_residual(episodes[7], arrays[7], cfg)
    assert issue is not None
    assert issue.issue_id == "l2.action_state_residual"
    assert issue.severity == "warn"
    assert issue.frame_end == 9
    assert check_action_state_residual(episodes[0], arrays[0], cfg) is None


def test_merge_issues_combines_ranges_and_details():
    rows = [
        make_issue("l2.joint_step", "warn", 1, 10, 12, 2, "action: 2 transitions"),
        make_issue("l2.joint_step", "warn", 1, 20, 21, 1, "state: 1 transition"),
        make_issue("l2.constant_channel", "warn", 1, 0, 29, 1, "dim 9"),
    ]
    merged = merge_issues(rows)
    assert len(merged) == 2
    step = next(i for i in merged if i.issue_id == "l2.joint_step")
    assert step.frame_start == 10
    assert step.frame_end == 21
    assert step.event_count == 3
    assert "action" in step.detail and "state" in step.detail


def test_suppress_step_covered_by_trajectory():
    step = make_issue("l2.joint_step", "warn", 1, 129, 148, 37, "step")
    trajectory = make_issue("l2.trajectory_deviation", "fail", 1, 110, 146, 37, "trajectory")
    other_step = make_issue("l2.joint_step", "warn", 2, 10, 20, 3, "other")
    kept = suppress_covered_step_issues([step, trajectory, other_step])
    assert step not in kept
    assert trajectory in kept
    assert other_step in kept

    # Non-overlapping segments stay separate.
    far_step = make_issue("l2.joint_step", "warn", 3, 200, 220, 3, "far")
    near = make_issue("l2.trajectory_deviation", "fail", 3, 10, 50, 5, "near")
    assert far_step in suppress_covered_step_issues([far_step, near])

    # A lower-severity trajectory does not suppress a higher-severity step.
    failing_step = make_issue("l2.joint_step", "fail", 4, 129, 148, 60, "failing step")
    warning_traj = make_issue("l2.trajectory_deviation", "warn", 4, 110, 146, 37, "warning")
    assert failing_step in suppress_covered_step_issues([failing_step, warning_traj])
