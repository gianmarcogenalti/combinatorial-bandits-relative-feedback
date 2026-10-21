"""Recursive max-min baseline for full-ranking combinatorial bandits.

Extracted without algorithm changes from the repository's original implementation
(revision dbd226e). The EWS adapter supplies checkpointing and exact score ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class FeedbackProtocol(Protocol):
    censored_ranks: dict[int, int]


class RoundResultProtocol(Protocol):
    pulled_arms: tuple[int, ...]
    feedback: FeedbackProtocol
    instantaneous_regret: float
    cumulative_regret: float
    t: int


@dataclass(frozen=True)
class AlgorithmState:
    """
    Lightweight public snapshot of the algorithm state.
    """

    t: int
    accepted: tuple[int, ...]
    unknown: tuple[int, ...]
    pivot: int
    completed: bool


class BaseAlgorithm:
    """
    Minimal interface for algorithms compatible with RelativeFeedbackCMABEnv.
    """

    def select_action(self) -> tuple[int, ...]:
        raise NotImplementedError

    def update(self, result: RoundResultProtocol) -> None:
        raise NotImplementedError

    def state(self) -> AlgorithmState:
        raise NotImplementedError


class RecMaxMinUCB(BaseAlgorithm):
    """
    Rec-MaxMin-UCB baseline from Saha and Gopalan.

    This is the baseline for Top-k-regret with full Top-k-ranking feedback.

    Important
    ---------
    The faithful version requires m == k, because the original Rec-MaxMin-UCB
    update uses a full ranking of the k pulled arms.

    Arms are 0-indexed: {0, ..., n - 1}.
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        m: Optional[int] = None,
        alpha: float = 0.51,
        seed: Optional[int] = None,
        require_full_ranking: bool = True,
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2.")
        if not (1 <= k <= n):
            raise ValueError("Require 1 <= k <= n.")
        if alpha <= 0.5:
            raise ValueError("Rec-MaxMin-UCB requires alpha > 0.5.")

        if m is not None and require_full_ranking and m != k:
            raise ValueError(
                "Faithful Rec-MaxMin-UCB requires full ranking feedback, so m must equal k. "
                "Run the environment with m=k, or set require_full_ranking=False for a "
                "top-m rank-breaking variant."
            )

        self.n = int(n)
        self.k = int(k)
        self.m = int(m) if m is not None else int(k)
        self.alpha = float(alpha)
        self.require_full_ranking = bool(require_full_ranking)

        self.rng = np.random.default_rng(seed)

        self.t = 0

        # W[i, j] = number of rank-broken observations where i beat j.
        self.W = np.zeros((self.n, self.n), dtype=float)

        # Ordered slots B_t(1), ..., B_t(k).
        # Each slot is represented as a set to stay close to the pseudocode,
        # though in normal operation each slot is empty or singleton.
        self.B_slots: list[set[int]] = [set() for _ in range(self.k)]

        self._last_action: Optional[tuple[int, ...]] = None

    def select_action(self) -> tuple[int, ...]:
        """
        Build the size-k subset S_t using the recursive max-min rule.
        """
        round_number = self.t + 1
        U = self._ucb_matrix(round_number)

        selected: set[int] = set()
        pool: set[int] = set(range(self.n))
        new_B_slots: list[set[int]] = [set() for _ in range(self.k)]

        exited_early = False

        # Recursive construction for slots 1, ..., k-1.
        for h in range(self.k - 1):
            C = self._condorcet_candidates(U, pool)

            intersection = C & self.B_slots[h]

            if len(intersection) > 0:
                chosen = self._choose_one(intersection, U=U, selected=selected)
                new_B_slots[h] = {chosen}
                selected.add(chosen)
                pool.discard(chosen)
                continue

            # No previous candidate survived for this slot.
            if len(C) > 0:
                chosen_slot = self._build_s(
                    U=U,
                    selected=selected,
                    pool=pool,
                    num_to_draw=1,
                    preferred_pool=C,
                )
                if len(chosen_slot) > 0:
                    chosen = chosen_slot[0]
                    new_B_slots[h] = {chosen}
                    selected.add(chosen)
                    pool.discard(chosen)

            # Fill the rest of the played set and stop the recursion.
            remaining = self.k - len(selected)
            fill = self._build_s(
                U=U,
                selected=selected,
                pool=pool,
                num_to_draw=remaining,
            )
            selected.update(fill)
            pool.difference_update(fill)

            exited_early = True
            break

        # Final slot k, only if the recursion did not exit early.
        if not exited_early:
            C = self._condorcet_candidates(U, pool)

            if len(C) == 1:
                chosen = next(iter(C))
                new_B_slots[self.k - 1] = {chosen}
                selected.add(chosen)
                pool.discard(chosen)
            else:
                chosen_final = self._build_s(
                    U=U,
                    selected=selected,
                    pool=pool,
                    num_to_draw=1,
                )
                selected.update(chosen_final)
                pool.difference_update(chosen_final)

        # Defensive fallback: ensure exactly k distinct arms.
        if len(selected) < self.k:
            fill = self._build_s(
                U=U,
                selected=selected,
                pool=set(range(self.n)) - selected,
                num_to_draw=self.k - len(selected),
            )
            selected.update(fill)

        if len(selected) > self.k:
            selected = set(
                self._choose_many_by_maxmin(
                    candidates=selected,
                    U=U,
                    selected_context=set(),
                    num_to_choose=self.k,
                )
            )

        action = tuple(sorted(selected))

        if len(action) != self.k:
            raise RuntimeError(
                f"RecMaxMinUCB selected {len(action)} arms instead of k={self.k}."
            )

        self.B_slots = new_B_slots
        self._last_action = action

        return action

    def update(self, result: RoundResultProtocol) -> None:
        """
        Rank-breaking update of W from the observed ranking.
        """
        pulled = tuple(int(i) for i in result.pulled_arms)
        ranks = dict(result.feedback.censored_ranks)

        if len(pulled) != self.k:
            raise ValueError(f"Expected exactly k={self.k} pulled arms.")
        if set(pulled) != set(ranks.keys()):
            raise ValueError("Feedback ranks must be available for every pulled arm.")

        if self.require_full_ranking:
            observed_ranks = set(ranks.values())
            required_ranks = set(range(1, self.k + 1))
            if observed_ranks != required_ranks:
                raise ValueError(
                    "Faithful Rec-MaxMin-UCB requires a full ranking of the k pulled arms. "
                    "Run the environment with m=k."
                )

        pulled_set = set(pulled)

        # Use every arm whose rank is actually observed.
        # If m=k, this is the full ranking.
        ranked_arms = [
            arm
            for arm, rank in sorted(ranks.items(), key=lambda item: item[1])
            if rank <= self.m
        ]

        prefix: set[int] = set()

        for winner in ranked_arms:
            prefix.add(winner)

            # Rank-breaking:
            # the item at this rank beats every item not ranked above it.
            for loser in pulled_set - prefix:
                self.W[winner, loser] += 1.0

        self.t += 1

    def state(self) -> AlgorithmState:
        selected_slots = self._flatten_B_slots()

        return AlgorithmState(
            t=self.t,
            accepted=tuple(sorted(selected_slots)),
            unknown=tuple(sorted(set(range(self.n)) - selected_slots)),
            pivot=-1,
            completed=len(selected_slots) == self.k,
        )

    def diagnostic_tables(self) -> dict[str, np.ndarray]:
        """
        Return pairwise counts, empirical preferences, and UCBs.
        """
        N = self.W + self.W.T
        P_hat = self._p_hat_matrix()
        U = self._ucb_matrix(max(self.t, 1))

        return {
            "W": self.W.copy(),
            "N": N.copy(),
            "P_hat": P_hat.copy(),
            "U": U.copy(),
        }

    def _flatten_B_slots(self) -> set[int]:
        out: set[int] = set()
        for slot in self.B_slots:
            out.update(slot)
        return out

    def _p_hat_matrix(self) -> np.ndarray:
        N = self.W + self.W.T

        P_hat = np.full((self.n, self.n), 0.5, dtype=float)

        mask = N > 0
        P_hat[mask] = self.W[mask] / N[mask]

        np.fill_diagonal(P_hat, 0.5)

        return P_hat

    def _ucb_matrix(self, round_number: int) -> np.ndarray:
        """
        u_ij = p_hat_ij + sqrt(alpha log(t) / n_ij)

        If n_ij = 0, set u_ij = +inf for i != j.
        """
        t = max(int(round_number), 1)

        N = self.W + self.W.T
        P_hat = self._p_hat_matrix()

        U = np.empty((self.n, self.n), dtype=float)

        log_t = np.log(t)

        positive = N > 0
        U[positive] = P_hat[positive] + np.sqrt(self.alpha * log_t / N[positive])
        U[~positive] = np.inf

        np.fill_diagonal(U, 0.5)

        return U

    def _condorcet_candidates(
        self,
        U: np.ndarray,
        pool: set[int],
    ) -> set[int]:
        """
        C = {i in pool : u_ij > 1/2 for all j in pool \\ {i}}.
        """
        candidates: set[int] = set()

        for i in pool:
            is_candidate = True
            for j in pool:
                if i == j:
                    continue
                if not (U[i, j] > 0.5):
                    is_candidate = False
                    break

            if is_candidate:
                candidates.add(i)

        return candidates

    def _build_s(
        self,
        *,
        U: np.ndarray,
        selected: set[int],
        pool: set[int],
        num_to_draw: int,
        preferred_pool: Optional[set[int]] = None,
    ) -> tuple[int, ...]:
        """
        Robust implementation of the paper's build_S max-min subroutine.

        It returns exactly num_to_draw arms whenever enough arms are available.
        """
        if num_to_draw <= 0:
            return tuple()

        available = set(pool) - set(selected)

        if preferred_pool is not None:
            preferred_available = set(preferred_pool) & available
        else:
            preferred_available = set()

        chosen: list[int] = []
        selected_context = set(selected)

        # First use preferred candidates, if supplied.
        while len(chosen) < num_to_draw and preferred_available:
            arm = self._choose_one(
                preferred_available,
                U=U,
                selected=selected_context,
            )
            chosen.append(arm)
            selected_context.add(arm)
            preferred_available.discard(arm)
            available.discard(arm)

        # Then repeatedly add current Condorcet candidates if they exist.
        while len(chosen) < num_to_draw:
            C = self._condorcet_candidates(U, available)

            if len(C) == 0:
                break

            arm = self._choose_one(
                C,
                U=U,
                selected=selected_context,
            )
            chosen.append(arm)
            selected_context.add(arm)
            available.discard(arm)

        # Finally use max-min completion.
        while len(chosen) < num_to_draw:
            if not available:
                break

            arm = self._choose_one(
                available,
                U=U,
                selected=selected_context,
            )
            chosen.append(arm)
            selected_context.add(arm)
            available.discard(arm)

        return tuple(chosen)

    def _choose_one(
        self,
        candidates: set[int],
        *,
        U: np.ndarray,
        selected: set[int],
    ) -> int:
        """
        Choose one arm using the max-min rule.

        If selected is empty, break ties uniformly at random.
        Otherwise choose argmax_c min_{i in selected} u_{c,i}.
        """
        if not candidates:
            raise ValueError("Cannot choose from an empty candidate set.")

        if len(selected) == 0:
            return int(self.rng.choice(sorted(candidates)))

        scores = {}
        for c in candidates:
            scores[c] = min(float(U[c, i]) for i in selected)

        max_score = max(scores.values())

        best = [
            arm
            for arm, score in scores.items()
            if np.isclose(score, max_score)
        ]

        return int(self.rng.choice(sorted(best)))

    def _choose_many_by_maxmin(
        self,
        *,
        candidates: set[int],
        U: np.ndarray,
        selected_context: set[int],
        num_to_choose: int,
    ) -> tuple[int, ...]:
        chosen: list[int] = []
        available = set(candidates)
        context = set(selected_context)

        while len(chosen) < num_to_choose and available:
            arm = self._choose_one(
                available,
                U=U,
                selected=context,
            )
            chosen.append(arm)
            context.add(arm)
            available.discard(arm)

        return tuple(chosen)
