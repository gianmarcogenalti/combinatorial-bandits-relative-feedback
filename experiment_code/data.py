"""Plackett-Luce feedback; expected rewards remain hidden from the learner."""

import numpy as np
from experiments_wo_stress import Feedback, Instance


class RankingBandit:
    """Save fixed preferences and sample top-m feedback by an exponential race."""

    supports_extension = True

    @classmethod
    def create_instance(cls, *, rng, theta, k, m, mnl=False, theta0=1.0):
        theta = np.asarray(theta, dtype=float)
        if (
            theta.ndim != 1
            or not np.isfinite(theta).all()
            or np.any(theta <= 0)
            or not 1 <= m <= k <= len(theta)
            or theta0 <= 0
            or not np.isfinite(theta0)
        ):
            raise ValueError("Invalid ranking bandit instance")
        if mnl and m != 1:
            raise ValueError("MNL uses winner feedback")
        return Instance(
            kind="pl_ranking",
            metadata={"k": k, "m": m, "mnl": mnl, "theta0": theta0},
            arrays={"theta": theta},
        )

    def __init__(self, *, rng, instance, theta, k, m, mnl=False, theta0=1.0):
        self.rng, self.theta, self.k, self.m, self.mnl = (
            rng,
            instance.arrays["theta"],
            k,
            m,
            mnl,
        )
        self.weights = np.append(self.theta, theta0) if mnl else self.theta

    def generate(self, action):
        action = np.asarray(action, dtype=np.int64)
        if (
            len(action) != self.k
            or len(np.unique(action)) != self.k
            or np.any(action < 0)
            or np.any(action >= len(self.theta))
        ):
            raise ValueError("Action must contain k distinct real arms")
        items = np.append(action, len(self.theta)) if self.mnl else action
        race = self.rng.exponential(1 / self.weights[items])
        order = np.argsort(race)[: self.m]
        ranks = np.full(len(self.weights), min(len(items), self.m + 1), dtype=np.int64)
        ranks[items[order]] = np.arange(1, len(order) + 1)
        return Feedback(ranks, {"action": action})

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Unexpected environment state")
