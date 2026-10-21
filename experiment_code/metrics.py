"""Compute mean-preference top-k regret from every saved action."""

import numpy as np
from experiments_wo_stress import MetricResult


class TopKRegret:
    """Regret is (sum of top-k theta minus sum of played theta) / k, also for MNL."""

    required_fields = ("step", "action")
    required_instance_fields = ("theta",)
    requires_complete_trajectory = True

    def __init__(self, points=200):
        if points < 2:
            raise ValueError("At least two metric points are required")
        self.points = int(points)

    def compute(self, result):
        theta = result.instance.arrays["theta"]
        actions = result["action"].astype(np.int64)
        k = actions.shape[1]
        optimal = np.sort(theta)[-k:].mean()
        regret = np.maximum(0, optimal - theta[actions].mean(axis=1))
        cumulative = np.cumsum(regret)
        indices = np.unique(
            np.linspace(
                0, len(cumulative) - 1, min(self.points, len(cumulative)), dtype=int
            )
        )
        return MetricResult(x=result["step"][indices], values=cumulative[indices])
