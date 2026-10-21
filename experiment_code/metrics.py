"""Compute mean-preference top-k regret from every saved action."""

import numpy as np
from experiments_wo_stress import MetricResult


class TopKRegret:
    """Regret is (sum of top-k theta minus sum of played theta) / k, also for MNL."""

    required_fields = ("step", "action")
    required_instance_fields = ("theta",)
    requires_complete_trajectory = True

    def compute(self, result):
        theta = result.instance.arrays["theta"]
        actions = result["action"].astype(np.int64)
        k = actions.shape[1]
        optimal = np.sort(theta)[-k:].mean()
        regret = np.maximum(0, optimal - theta[actions].mean(axis=1))
        return MetricResult(x=result["step"], values=np.cumsum(regret))
