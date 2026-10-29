"""Issue model, severity/action mapping, range merging and sorting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .config import Config


_SEVERITY_RANK = {"fail": 3, "warn": 2, "info": 1}

_DROP_CANDIDATE_IDS = {
    "l0.missing_referenced_file",
    "l0.data_row_count",
    "l0.episode_index",
    "l0.global_index",
    "l0.file_row_count",
    "l0.empty_file",
    "l2.nan_inf",
}


@dataclass(frozen=True)
class Issue:
    issue_id: str
    severity: str
    episode_index: int | None
    frame_start: int | None
    frame_end: int | None
    event_count: int
    detail: str
    action: str


def action_for(issue_id: str, severity: str) -> str:
    if severity == "fail":
        return "drop_candidate" if issue_id in _DROP_CANDIDATE_IDS else "review"
    if severity == "warn":
        return "review"
    return "note"


def make_issue(
    issue_id: str,
    severity: str,
    episode_index: int | None,
    frame_start: int | None,
    frame_end: int | None,
    event_count: int,
    detail: str,
) -> Issue:
    if severity not in _SEVERITY_RANK:
        raise ValueError(f"invalid severity: {severity!r}")
    return Issue(
        issue_id=issue_id,
        severity=severity,
        episode_index=episode_index,
        frame_start=frame_start,
        frame_end=frame_end,
        event_count=int(event_count),
        detail=detail,
        action=action_for(issue_id, severity),
    )


def dataset_issue(issue_id: str, severity: str, detail: str) -> Issue:
    return make_issue(issue_id, severity, None, None, None, 1, detail)


def frame_span(frames: Iterable[int], length: int) -> tuple[int, int, int] | None:
    """Return (start, end, unique_frame_count) for a set of frame indices."""
    arr = np.unique(np.asarray(list(frames), dtype=np.int64))
    if arr.size == 0:
        return None
    start = max(0, int(arr[0]))
    end = min(max(0, int(length) - 1), int(arr[-1]))
    if end < start:
        return None
    return start, end, int(arr.size)


def transition_frames(transitions: Iterable[int], length: int) -> list[int]:
    """Expand transition t -> [t, t+1] for reporting."""
    frames: set[int] = set()
    last = max(0, int(length) - 1)
    for t in transitions:
        t = int(t)
        frames.add(max(0, min(last, t)))
        frames.add(max(0, min(last, t + 1)))
    return sorted(frames)


def examples_text(frames: Iterable[int], limit: int) -> str:
    arr = np.unique(np.asarray(list(frames), dtype=np.int64))
    shown = [str(int(x)) for x in arr[:limit]]
    suffix = "" if arr.size <= limit else f", ... (+{arr.size - limit})"
    return ", ".join(shown) + suffix


def merge_issues(issues: Iterable[Issue]) -> list[Issue]:
    """Merge same (episode_index, issue_id) rows into one concise row."""
    groups: dict[tuple[int | None, str], list[Issue]] = {}
    for issue in issues:
        groups.setdefault((issue.episode_index, issue.issue_id), []).append(issue)

    merged: list[Issue] = []
    for (episode_index, issue_id), rows in groups.items():
        rows.sort(key=lambda r: r.severity, reverse=True)
        severity = rows[0].severity
        starts = [r.frame_start for r in rows if r.frame_start is not None]
        ends = [r.frame_end for r in rows if r.frame_end is not None]
        details: list[str] = []
        for row in rows:
            if row.detail and row.detail not in details:
                details.append(row.detail)
        merged.append(
            make_issue(
                issue_id=issue_id,
                severity=severity,
                episode_index=episode_index,
                frame_start=min(starts) if starts else None,
                frame_end=max(ends) if ends else None,
                event_count=sum(r.event_count for r in rows),
                detail="; ".join(details),
            )
        )
    return sort_issues(merged)


def severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(severity, 0)


def sort_issues(issues: Iterable[Issue]) -> list[Issue]:
    return sorted(
        issues,
        key=lambda i: (
            -severity_rank(i.severity),
            i.episode_index if i.episode_index is not None else -1,
            i.issue_id,
            i.frame_start if i.frame_start is not None else -1,
        ),
    )


def worst_severity(issues: Iterable[Issue]) -> str:
    rank = 0
    result = "info"
    for issue in issues:
        r = severity_rank(issue.severity)
        if r > rank:
            rank = r
            result = issue.severity
    return result


def replace_severity(issue: Issue, severity: str) -> Issue:
    return replace(issue, severity=severity, action=action_for(issue.issue_id, severity))


def suppress_covered_step_issues(issues: Iterable[Issue], max_gap: int = 5) -> list[Issue]:
    """Drop a joint-step warning when a trajectory deviation covers it.

    The trajectory row is the higher-level explanation of the same motion, so
    reporting both would duplicate the same episode problem.
    """
    issue_list = list(issues)
    trajectories = [
        issue
        for issue in issue_list
        if issue.issue_id == "l2.trajectory_deviation" and issue.episode_index is not None
    ]
    kept: list[Issue] = []
    for issue in issue_list:
        if issue.issue_id != "l2.joint_step":
            kept.append(issue)
            continue
        covered = False
        for trajectory in trajectories:
            if trajectory.episode_index != issue.episode_index:
                continue
            if severity_rank(trajectory.severity) < severity_rank(issue.severity):
                continue
            if (
                issue.frame_start is None
                or issue.frame_end is None
                or trajectory.frame_start is None
                or trajectory.frame_end is None
            ):
                continue
            if not (
                issue.frame_end + max_gap < trajectory.frame_start
                or trajectory.frame_end + max_gap < issue.frame_start
            ):
                covered = True
                break
        if not covered:
            kept.append(issue)
    return kept
