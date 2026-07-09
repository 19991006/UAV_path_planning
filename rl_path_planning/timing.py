"""Wall-clock timing utilities for assignment and inference measurement."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TimingRecorder:
    """Records per-step timing data during evaluation."""

    assign_times: List[float] = field(default_factory=list)

    def record_assign(self, elapsed: float) -> None:
        self.assign_times.append(elapsed)

    def reset(self) -> None:
        self.assign_times.clear()

    def summary(self) -> dict:
        if not self.assign_times:
            return {}
        arr = np.array(self.assign_times)
        return {
            "assign_mean_ms": float(arr.mean() * 1000),
            "assign_std_ms": float(arr.std() * 1000),
            "assign_min_ms": float(arr.min() * 1000),
            "assign_max_ms": float(arr.max() * 1000),
        }
