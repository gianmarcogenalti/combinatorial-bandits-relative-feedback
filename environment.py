# environment.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TopMFeedback:
    """
    Top-m censored Plackett-Luce feedback.

    Attributes
    ----------
    full_ranking:
        Tuple of all pulled arms ordered from best rank to worst rank.
        This is returned mainly for debugging; algorithms should usually use
        `censored_ranks`.

    censored_ranks:
        Dictionary mapping each pulled arm i to R_m(i), where ranks are 1-indexed.
        Arms outside the observed top-m have rank m + 1.

    top_m:
        Tuple of arms appearing in the observed top-m ranking.
    """

    full_ranking: tuple[int, ...]
    censored_ranks: dict[int, int]
    top_m: tuple[int, ...]


@dataclass(frozen=True)
class RoundResult:
    """
    Output of one environment step.
    """

    pulled_arms: tuple[int, ...]
    feedback: TopMFeedback
    expected_reward: float
    instantaneous_regret: float
    cumulative_regret: float
    t: int


class RelativeFeedbackCMABEnv:
    """
    Combinatorial bandit environment with Plackett-Luce relative feedback.

    Setting
    -------
    - There are n arms with positive PL parameters theta.
    - At every round, the learner pulls exactly k distinct arms.
    - The environment samples a full Plackett-Luce ranking of the pulled arms.
    - The learner observes only the top-m censored ranking:
        R_m(i) = rank(i) if rank(i) <= m, else m + 1.
    - Expected reward is:
        g_theta^{(k)}(S) = (1 / k) * sum_{i in S} theta_i.
    - Regret is measured against the top-k arms by theta.

    Notes
    -----
    Arms are 0-indexed: {0, ..., n - 1}.
    """

    def __init__(
        self,
        theta: Sequence[float],
        k: int,
        m: int,
        *,
        seed: Optional[int] = None,
        normalize_theta: bool = True,
        require_unique_top_k: bool = False,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float).copy()

        if self.theta.ndim != 1:
            raise ValueError("theta must be a one-dimensional sequence.")
        if len(self.theta) < 2:
            raise ValueError("The number of arms n must be at least 2.")
        if np.any(self.theta <= 0):
            raise ValueError("All PL parameters theta_i must be strictly positive.")

        self.n = int(len(self.theta))
        self.k = int(k)
        self.m = int(m)

        if not (1 <= self.m <= self.k <= self.n):
            raise ValueError("Parameters must satisfy 1 <= m <= k <= n.")

        if normalize_theta:
            self.theta /= np.max(self.theta)

        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.cumulative_regret = 0.0

        self._top_k_arms = tuple(np.argsort(-self.theta, kind="stable")[: self.k])
        self._optimal_reward = self.reward(self._top_k_arms)

        if require_unique_top_k:
            sorted_theta = np.sort(self.theta)[::-1]
            if np.isclose(sorted_theta[self.k - 1], sorted_theta[self.k]):
                raise ValueError(
                    "The top-k set is not unique: theta_(k) equals theta_(k+1)."
                )

    @property
    def top_k_arms(self) -> tuple[int, ...]:
        """
        The optimal top-k arm set under theta.
        """
        return self._top_k_arms

    @property
    def optimal_reward(self) -> float:
        """
        Reward of the optimal top-k set.
        """
        return self._optimal_reward

    def reset(self, *, seed: Optional[int] = None) -> None:
        """
        Reset time and cumulative regret.

        If seed is provided, also reset the RNG.
        """
        self.t = 0
        self.cumulative_regret = 0.0

        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def reward(self, arms: Iterable[int]) -> float:
        """
        Expected reward g_theta^{(k)}(S) = average theta over selected arms.
        """
        selected = self._validate_arms(arms, require_size_k=False)
        return float(np.mean(self.theta[list(selected)]))

    def regret(self, arms: Iterable[int]) -> float:
        """
        Instantaneous top-k regret of pulling the given arm set.
        """
        selected = self._validate_arms(arms, require_size_k=True)
        return float(self.optimal_reward - self.reward(selected))

    def step(self, pulled_arms: Iterable[int]) -> RoundResult:
        """
        Pull a size-k set of arms and receive top-m relative feedback.

        Parameters
        ----------
        pulled_arms:
            Iterable of exactly k distinct arm indices.

        Returns
        -------
        RoundResult
            Feedback, expected reward, instantaneous regret, and cumulative regret.
        """
        selected = self._validate_arms(pulled_arms, require_size_k=True)

        feedback = self.sample_top_m_feedback(selected)
        expected_reward = self.reward(selected)
        instantaneous_regret = self.optimal_reward - expected_reward

        self.t += 1
        self.cumulative_regret += instantaneous_regret

        return RoundResult(
            pulled_arms=selected,
            feedback=feedback,
            expected_reward=expected_reward,
            instantaneous_regret=float(instantaneous_regret),
            cumulative_regret=float(self.cumulative_regret),
            t=self.t,
        )

    def sample_pl_ranking(self, arms: Iterable[int]) -> tuple[int, ...]:
        """
        Sample a full Plackett-Luce ranking over the provided arms.

        Uses the exponential-race representation:
        if E_i ~ Exp(theta_i), then sorting by increasing E_i gives a PL ranking.
        """
        selected = self._validate_arms(arms, require_size_k=False)
        selected_array = np.asarray(selected, dtype=int)

        rates = self.theta[selected_array]
        race_times = self.rng.exponential(scale=1.0 / rates)

        ranking = selected_array[np.argsort(race_times, kind="stable")]
        return tuple(int(i) for i in ranking)

    def sample_top_m_feedback(self, arms: Iterable[int]) -> TopMFeedback:
        """
        Sample the observed top-m censored ranking for the pulled arms.
        """
        selected = self._validate_arms(arms, require_size_k=True)
        full_ranking = self.sample_pl_ranking(selected)

        censored_ranks: dict[int, int] = {}
        for rank, arm in enumerate(full_ranking, start=1):
            censored_ranks[arm] = rank if rank <= self.m else self.m + 1

        return TopMFeedback(
            full_ranking=full_ranking,
            censored_ranks=censored_ranks,
            top_m=full_ranking[: self.m],
        )

    def theta_gap_to_boundary(self) -> float:
        """
        Return theta_(k) - theta_(k+1), useful for diagnostics.
        """
        sorted_theta = np.sort(self.theta)[::-1]
        return float(sorted_theta[self.k - 1] - sorted_theta[self.k])

    def arm_is_optimal(self, arm: int) -> bool:
        """
        Check whether an arm belongs to the true top-k set.
        """
        self._validate_single_arm(arm)
        return int(arm) in set(self.top_k_arms)

    def _validate_arms(
        self,
        arms: Iterable[int],
        *,
        require_size_k: bool,
    ) -> tuple[int, ...]:
        selected = tuple(int(a) for a in arms)

        if require_size_k and len(selected) != self.k:
            raise ValueError(f"Expected exactly k={self.k} arms, got {len(selected)}.")

        if len(set(selected)) != len(selected):
            raise ValueError("Pulled arms must be distinct.")

        for arm in selected:
            self._validate_single_arm(arm)

        return selected

    def _validate_single_arm(self, arm: int) -> None:
        if not (0 <= int(arm) < self.n):
            raise ValueError(f"Arm index {arm} is out of bounds for n={self.n}.")


def make_geometric_theta(
    n: int,
    *,
    ratio: float = 0.9,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    Convenience generator for decreasing PL parameters.

    Example:
        n=5, ratio=0.8 -> [1.0, 0.8, 0.64, 0.512, 0.4096]
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (0 < ratio <= 1):
        raise ValueError("ratio must be in (0, 1].")
    if max_value <= 0:
        raise ValueError("max_value must be positive.")

    return max_value * ratio ** np.arange(n)


def make_gap_theta(
    n: int,
    k: int,
    *,
    top_value: float = 1.0,
    boundary_gap: float = 0.1,
    tail_value: Optional[float] = None,
) -> np.ndarray:
    """
    Convenience generator with a controlled gap between top-k and non-top-k arms.

    The first k arms have value top_value.
    The remaining n-k arms have value top_value - boundary_gap by default.

    This is useful for sanity-checking algorithms, although it creates ties inside
    the top-k and inside the non-top-k groups.
    """
    if not (1 <= k < n):
        raise ValueError("Require 1 <= k < n.")
    if top_value <= 0:
        raise ValueError("top_value must be positive.")
    if not (0 < boundary_gap < top_value):
        raise ValueError("boundary_gap must be in (0, top_value).")

    if tail_value is None:
        tail_value = top_value - boundary_gap
    if tail_value <= 0:
        raise ValueError("tail_value must be positive.")

    theta = np.empty(n, dtype=float)
    theta[:k] = top_value
    theta[k:] = tail_value
    return theta