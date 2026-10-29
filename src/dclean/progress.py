"""Small dependency-free progress reporting to stderr."""

from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressBar:
    """A ``\\r`` progress bar; disabled when stderr is not a TTY."""

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        min_interval_s: float = 0.2,
    ) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.enabled = bool(enabled) and (stream or sys.stderr).isatty()
        self.stream = stream or sys.stderr
        self.min_interval_s = min_interval_s
        self.done = 0
        self._last = 0.0
        self._visible = False

    def update(self, n: int = 1) -> None:
        self.done += n
        if not self.enabled:
            return
        now = time.monotonic()
        if self.done < self.total and now - self._last < self.min_interval_s:
            return
        self._last = now
        pct = 100.0 * self.done / self.total
        print(
            f"\r{self.desc}: {self.done}/{self.total} ({pct:5.1f}%)",
            end="",
            file=self.stream,
            flush=True,
        )
        self._visible = True

    def close(self) -> None:
        if self.enabled and self._visible:
            print(file=self.stream, flush=True)
        self._visible = False


class Reporter:
    """Stage lines plus optional progress bars."""

    def __init__(self, total_stages: int = 6, *, enabled: bool = True) -> None:
        self.total_stages = total_stages
        self.enabled = enabled
        self._stage = 0

    def stage(self, name: str) -> None:
        self._stage += 1
        print(
            f"[{self._stage}/{self.total_stages}] {name}...",
            file=sys.stderr,
            flush=True,
        )

    def info(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def bar(self, total: int, desc: str) -> ProgressBar:
        return ProgressBar(total, desc, enabled=self.enabled)
