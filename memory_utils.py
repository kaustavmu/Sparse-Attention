"""
Lightweight utilities for tracking process memory usage during experiments.

Designed so training scripts can log peak RSS and checkpoints without adding
extra third-party dependencies. If `psutil` is available we use it for precise
RSS sampling, otherwise we fall back to Python's built-in `resource` module.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


def _format_bytes(num: float) -> str:
    """Convert byte counts into a human-friendly string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f} {unit}"
        num /= 1024.0
    return f"{num:3.2f} PB"


def _rss_bytes() -> int:
    """Best-effort retrieval of the current resident set size."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        if sys.platform.startswith("linux"):
            return usage * 1024
        return usage

    # Fallback when neither psutil nor resource is available.
    return 0


@dataclass
class MemoryTracker:
    """Simple memory tracker that records RSS checkpoints and peak usage."""

    label: str = "run"
    checkpoints: List[Tuple[str, int]] = field(default_factory=list)
    peak_rss: int = 0

    def checkpoint(self, tag: str) -> int:
        """Record the current RSS with an associated tag."""
        current = _rss_bytes()
        self.peak_rss = max(self.peak_rss, current)
        self.checkpoints.append((tag, current))
        print(
            f"[Memory][{self.label}] {tag}: "
            f"{_format_bytes(current)} (peak {_format_bytes(self.peak_rss)})"
        )
        return current

    def report(self) -> None:
        """Print a final summary of peak RSS and the checkpoint table."""
        if not self.checkpoints:
            self.checkpoint("initial")

        print(f"[Memory][{self.label}] Summary:")
        for tag, value in self.checkpoints:
            print(f"  - {tag:<20} {_format_bytes(value)}")
        print(f"[Memory][{self.label}] Peak RSS: {_format_bytes(self.peak_rss)}")

