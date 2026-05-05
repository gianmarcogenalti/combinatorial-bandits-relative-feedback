# algorithms.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

import numpy as np


class FeedbackProtocol(Protocol):
    censored_ranks: dict[int, int]


class RoundResultProtocol(Protocol):
    pulled_arms: tuple[int, ...]
    feedback: FeedbackProtocol
    instantaneous_regret: float
    cumulative_regret: float
    t: int


def default_beta(delta: float, n: int, T: int) -> Callable[[float], float]:
    """
    Confidence bonus used in the draft:

        beta_{delta,n,T}(N) =
            +inf                         if N = 0
            sqrt(log(2 n T / delta) / N) if N > 0
            0                            if N = +inf

    The diagonal counts N_i^(i) are initialized to +inf.
    """

    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")
    if n < 2:
        raise ValueError("n must be at least 2.")
    if T < 1:
        raise ValueError("T must be at least 1.")

    log_term = np.log((2.0 * n * T) / delta)

    def beta(N: float) -> float:
        if np.isposinf(N):
            return 0.0
        if N <= 0:
            return float("inf")
        return float(0.25*np.sqrt(log_term / N))

    return beta


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


class TopKUIF(BaseAlgorithm):
    """
    TOP-k-UIF with safer finite-time behavior.

    Main differences from the previous implementation:
    - Does not silently complete when |A| = k but U is nonempty.
    - Applies accept/reject decisions in batches rather than sorted sequential order.
    - Tracks invariant violations for debugging.
    - Keeps pivot-set updates restricted to pivots that are actually pulled.
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        m: int,
        T: int,
        delta: float = 0.05,
        seed: Optional[int] = None,
        beta_fn: Optional[Callable[[float], float]] = None,
        soft_guards: bool = True,
        strict_invariants: bool = False,
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2.")
        if not (1 <= m <= k <= n):
            raise ValueError("Require 1 <= m <= k <= n.")
        if T < 1:
            raise ValueError("T must be at least 1.")
        if not (0 < delta <= 1):
            raise ValueError("delta must be in (0, 1].")

        self.n = int(n)
        self.k = int(k)
        self.m = int(m)
        self.T = int(T)
        self.delta = float(delta)
        self.soft_guards = bool(soft_guards)
        self.strict_invariants = bool(strict_invariants)

        self.rng = np.random.default_rng(seed)
        self.beta_fn = beta_fn if beta_fn is not None else default_beta(delta, n, T)

        self.t = 0

        self.accepted: set[int] = set()
        self.unknown: set[int] = set(range(self.n))
        self.pivot: int = int(self.rng.integers(self.n))

        self.mu_hat = np.full((self.n, self.n), 0.5, dtype=float)

        self.N = np.zeros((self.n, self.n), dtype=float)
        np.fill_diagonal(self.N, np.inf)

        self._last_action: Optional[tuple[int, ...]] = None
        self._completed = False
        self.invariant_violations: list[str] = []

    def select_action(self) -> tuple[int, ...]:
        """
        Select S_t.

        The theoretical action is:
            Pset = A_t union {pivot}
            S_exp ~ Uniform subsets of U_t \\ {pivot}
            S_t = Pset union S_exp

        This version keeps running even outside the good event, but records
        invariant violations rather than silently pretending everything is fine.
        """

        self._check_basic_invariants(context="before select_action")

        # True terminal condition expected by the theory.
        if len(self.accepted) == self.k and len(self.unknown) == 0:
            self._completed = True
            action = tuple(sorted(self.accepted))
            self._last_action = action
            return action

        # Suspicious finite-time state: accepted is full but unknown remains.
        if len(self.accepted) >= self.k and len(self.unknown) > 0:
            self._record_violation(
                "len(accepted) >= k while unknown is nonempty. "
                "This means the good-event terminal property failed."
            )

            # For regret evaluation, play accepted arms.
            # For debugging, this is intentionally recorded as not cleanly completed.
            action = tuple(sorted(self.accepted)[: self.k])
            self._last_action = action
            return action

        pivot_set = set(self.accepted) | {self.pivot}

        if len(pivot_set) > self.k:
            self._record_violation(
                f"pivot_set has size {len(pivot_set)} > k={self.k}."
            )
            action = tuple(sorted(self.accepted)[: self.k])
            self._last_action = action
            return action

        num_to_sample = self.k - len(pivot_set)

        candidates = sorted(self.unknown - {self.pivot})

        if len(candidates) < num_to_sample:
            self._record_violation(
                f"Not enough unknown candidates to sample: need {num_to_sample}, "
                f"have {len(candidates)}."
            )

            action_set = set(pivot_set) | set(candidates)
            remaining = [i for i in range(self.n) if i not in action_set]

            if len(remaining) < self.k - len(action_set):
                raise RuntimeError("Cannot construct a valid size-k action.")

            extra = self._sample_without_replacement(
                remaining,
                self.k - len(action_set),
            )
            action_set.update(extra)

        else:
            sampled = self._sample_without_replacement(candidates, num_to_sample)
            action_set = pivot_set | set(sampled)

        action = tuple(sorted(action_set))

        if len(action) != self.k:
            raise RuntimeError(
                f"Selected {len(action)} arms instead of k={self.k}: {action}"
            )

        self._last_action = action
        return action

    def update(self, result: RoundResultProtocol) -> None:
        pulled_arms = tuple(int(i) for i in result.pulled_arms)
        pulled_set = set(pulled_arms)
        censored_ranks = result.feedback.censored_ranks

        if len(pulled_arms) != self.k:
            raise ValueError(f"Expected result for k={self.k} pulled arms.")
        if pulled_set != set(censored_ranks.keys()):
            raise ValueError("Feedback ranks must be provided for all pulled arms.")

        self.t += 1

        old_accepted = set(self.accepted)
        old_unknown = set(self.unknown)
        old_pivot = int(self.pivot)

        # In the theoretical algorithm, pivot_set is always contained in the action.
        # Outside the good event, enforce that updates only use actually pulled pivots.
        pivot_set = (old_accepted | {old_pivot}) & pulled_set

        if len(pivot_set) == 0:
            self._record_violation(
                "No pivot from A union {pivot} was pulled; skipping RB update."
            )
        else:
            self._rb_update(
                pivot_set=pivot_set,
                pulled_arms=pulled_arms,
                censored_ranks=censored_ranks,
            )

        # COUNT-BA uses old A_t and old U_t.
        A_count, B_count = self._count_ba(
            pivot_set=(old_accepted | {old_pivot}),
            unknown_arms=sorted(old_unknown),
        )

        reject_candidates: set[int] = set()
        accept_candidates: set[int] = set()

        reject_threshold = self.k - len(old_accepted)
        accept_threshold = len(old_accepted | old_unknown) - self.k

        for i in old_unknown:
            if B_count[i] >= reject_threshold:
                reject_candidates.add(i)
            elif A_count[i] >= accept_threshold:
                accept_candidates.add(i)

        new_accepted = set(old_accepted)
        new_unknown = set(old_unknown)

        # Apply rejections first.
        if self.soft_guards:
            max_rejections = max(0, len(new_accepted | new_unknown) - self.k)
            if len(reject_candidates) > max_rejections:
                reject_candidates = set(
                    self.rng.choice(
                        sorted(reject_candidates),
                        size=max_rejections,
                        replace=False,
                    )
                )

        new_unknown -= reject_candidates

        # Apply acceptances, capped by remaining slots.
        accept_candidates &= new_unknown

        if self.soft_guards:
            available_slots = max(0, self.k - len(new_accepted))
            if len(accept_candidates) > available_slots:
                accept_candidates = set(
                    self.rng.choice(
                        sorted(accept_candidates),
                        size=available_slots,
                        replace=False,
                    )
                )

        new_accepted |= accept_candidates
        new_unknown -= accept_candidates

        new_pivot = old_pivot

        # Pivot update branch depends on old A_t, not updated A_{t+1}.
        if len(old_accepted) != 0:
            pivot_accept_candidates = {
                i for i in old_unknown
                if i in new_unknown and self.lcb(old_pivot, i) > 0.5
            }

            if self.soft_guards:
                available_slots = max(0, self.k - len(new_accepted))
                if len(pivot_accept_candidates) > available_slots:
                    pivot_accept_candidates = set(
                        self.rng.choice(
                            sorted(pivot_accept_candidates),
                            size=available_slots,
                            replace=False,
                        )
                    )

            new_accepted |= pivot_accept_candidates
            new_unknown -= pivot_accept_candidates

        else:
            candidate = self._new_pivot(
                pivot=old_pivot,
                unknown_arms=sorted(new_unknown),
            )
            if candidate is not None:
                new_pivot = candidate

        self.accepted = new_accepted
        self.unknown = new_unknown
        self.pivot = int(new_pivot)

        # Clean completion only.
        self._completed = len(self.accepted) == self.k and len(self.unknown) == 0

        self._check_basic_invariants(context="after update")

    def ucb(self, pivot: int, arm: int) -> float:
        return float(self.mu_hat[pivot, arm] + self.beta_fn(self.N[pivot, arm]))

    def lcb(self, pivot: int, arm: int) -> float:
        return float(self.mu_hat[pivot, arm] - self.beta_fn(self.N[pivot, arm]))

    def state(self) -> AlgorithmState:
        return AlgorithmState(
            t=self.t,
            accepted=tuple(sorted(self.accepted)),
            unknown=tuple(sorted(self.unknown)),
            pivot=int(self.pivot),
            completed=bool(self._completed),
        )

    def diagnostic_tables(self) -> dict[str, np.ndarray]:
        return {
            "mu_hat": self.mu_hat.copy(),
            "N": self.N.copy(),
        }

    def diagnostics(self) -> dict:
        return {
            "t": self.t,
            "accepted": tuple(sorted(self.accepted)),
            "unknown": tuple(sorted(self.unknown)),
            "pivot": self.pivot,
            "completed": self._completed,
            "num_invariant_violations": len(self.invariant_violations),
            "last_invariant_violations": self.invariant_violations[-10:],
        }

    def _rb_update(
        self,
        *,
        pivot_set: Iterable[int],
        pulled_arms: Iterable[int],
        censored_ranks: dict[int, int],
    ) -> None:
        for p in pivot_set:
            p = int(p)

            if p not in censored_ranks:
                self._record_violation(
                    f"Pivot {p} was not in censored_ranks; skipping."
                )
                continue

            rank_p = censored_ranks[p]

            for i in pulled_arms:
                i = int(i)
                rank_i = censored_ranks[i]

                if rank_p != rank_i:
                    old_N = self.N[p, i]
                    new_N = old_N + 1.0

                    indicator = 1.0 if rank_i < rank_p else 0.0

                    self.mu_hat[p, i] = (
                        ((new_N - 1.0) / new_N) * self.mu_hat[p, i]
                        + indicator / new_N
                    )
                    self.N[p, i] = new_N

    def _count_ba(
        self,
        *,
        pivot_set: Iterable[int],
        unknown_arms: Iterable[int],
    ) -> tuple[dict[int, int], dict[int, int]]:
        pivots = tuple(int(p) for p in pivot_set)
        unknown = tuple(int(i) for i in unknown_arms)

        A_count = {i: 0 for i in unknown}
        B_count = {i: 0 for i in unknown}

        for i in unknown:
            for j in unknown:
                i_loses_to_j = any(
                    self.ucb(p, i) < self.lcb(p, j)
                    for p in pivots
                )
                if i_loses_to_j:
                    B_count[i] += 1

                i_beats_j = any(
                    self.lcb(p, i) > self.ucb(p, j)
                    for p in pivots
                )
                if i_beats_j:
                    A_count[i] += 1

        return A_count, B_count

    def _new_pivot(
        self,
        *,
        pivot: int,
        unknown_arms: Iterable[int],
    ) -> Optional[int]:
        candidates = [
            int(i)
            for i in unknown_arms
            if self.lcb(int(pivot), int(i)) > 0.5
        ]

        if not candidates:
            return None

        scores = np.asarray([self.lcb(pivot, i) for i in candidates], dtype=float)
        max_score = np.max(scores)

        best = [
            arm
            for arm, score in zip(candidates, scores)
            if np.isclose(score, max_score)
        ]

        return int(self.rng.choice(best))

    def _sample_without_replacement(
        self,
        candidates: list[int],
        size: int,
    ) -> tuple[int, ...]:
        if size < 0:
            raise ValueError("Cannot sample a negative number of arms.")
        if size == 0:
            return tuple()
        if len(candidates) < size:
            raise ValueError(
                f"Cannot sample {size} items from only {len(candidates)} candidates."
            )

        sampled = self.rng.choice(candidates, size=size, replace=False)
        return tuple(int(i) for i in sampled)

    def _record_violation(self, message: str) -> None:
        full = f"t={self.t}: {message}"
        self.invariant_violations.append(full)

        if self.strict_invariants:
            raise RuntimeError(full)

    def _check_basic_invariants(self, *, context: str) -> None:
        if len(self.accepted) > self.k:
            self._record_violation(
                f"{context}: len(accepted)={len(self.accepted)} > k={self.k}."
            )

        if len(self.accepted & self.unknown) > 0:
            self._record_violation(
                f"{context}: accepted and unknown overlap: "
                f"{sorted(self.accepted & self.unknown)}."
            )

        if not (0 <= self.pivot < self.n):
            self._record_violation(
                f"{context}: pivot {self.pivot} outside [0, {self.n})."
            )


class UniformRandomTopK(BaseAlgorithm):
    """
    Simple baseline: pull k arms uniformly at random each round.
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        seed: Optional[int] = None,
    ) -> None:
        if not (1 <= k <= n):
            raise ValueError("Require 1 <= k <= n.")

        self.n = int(n)
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def select_action(self) -> tuple[int, ...]:
        action = self.rng.choice(self.n, size=self.k, replace=False)
        return tuple(sorted(int(i) for i in action))

    def update(self, result: RoundResultProtocol) -> None:
        self.t += 1

    def state(self) -> AlgorithmState:
        return AlgorithmState(
            t=self.t,
            accepted=tuple(),
            unknown=tuple(range(self.n)),
            pivot=-1,
            completed=False,
        )
    
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
    
# ============================================================
# MNL bandits with top-k objective
# ============================================================

class MNLTopKUIF(BaseAlgorithm):
    """
    MNL-TOP-k-UIF for MNL bandits with top-k objective.

    Real arms are 0, ..., n - 1.
    The no-choice fictitious item is represented internally as arm n.

    The environment must return feedback.censored_ranks containing ranks for
    every pulled real arm plus the no-choice item n. With MNL winner feedback:
        winner rank = 1
        all other items, including no-choice if not selected, have rank = 2.
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        T: int,
        delta: float = 0.05,
        seed: Optional[int] = None,
        beta_fn: Optional[Callable[[float], float]] = None,
        beta_scale: float = 1.0,
        soft_guards: bool = True,
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2.")
        if not (1 <= k <= n):
            raise ValueError("Require 1 <= k <= n.")
        if T < 1:
            raise ValueError("T must be at least 1.")

        self.n = int(n)
        self.k = int(k)
        self.T = int(T)
        self.delta = float(delta)

        self.no_choice = self.n
        self.dim = self.n + 1

        self.rng = np.random.default_rng(seed)
        self.soft_guards = bool(soft_guards)

        if beta_fn is not None:
            self.beta_fn = beta_fn
        else:
            base_beta = default_beta(delta, n + 1, T)

            def scaled_beta(N: float) -> float:
                return beta_scale * base_beta(N)

            self.beta_fn = scaled_beta

        self.beta_scale = float(beta_scale)

        self.t = 0
        self.accepted: set[int] = set()
        self.unknown: set[int] = set(range(self.n))

        # MNL-TOP-k-UIF starts with the no-choice item as pivot.
        self.pivot: int = self.no_choice

        self.mu_hat = np.full((self.dim, self.dim), 0.5, dtype=float)
        self.N = np.zeros((self.dim, self.dim), dtype=float)
        np.fill_diagonal(self.N, np.inf)

        self._last_action: Optional[tuple[int, ...]] = None
        self._completed = False
        self.invariant_violations: list[str] = []

    def select_action(self) -> tuple[int, ...]:
        if len(self.accepted) == self.k and len(self.unknown) == 0:
            self._completed = True
            action = tuple(sorted(self.accepted))
            self._last_action = action
            return action

        if len(self.accepted) >= self.k:
            action = tuple(sorted(self.accepted)[: self.k])
            self._last_action = action
            return action

        pivot_set = set(self.accepted) | {self.pivot}
        real_pivots = {p for p in pivot_set if p != self.no_choice}

        # Algorithm 3:
        # |S_exp| = k - |P_t| + I[no-choice in P_t]
        num_to_sample = self.k - len(pivot_set) + int(self.no_choice in pivot_set)

        candidates = set(self.unknown)
        if self.pivot != self.no_choice:
            candidates.discard(self.pivot)
        candidates = sorted(candidates)

        if len(candidates) >= num_to_sample:
            sampled = self._sample_without_replacement(candidates, num_to_sample)
            action_set = real_pivots | set(sampled)
        else:
            self.invariant_violations.append(
                f"t={self.t}: not enough candidates, need {num_to_sample}, "
                f"have {len(candidates)}."
            )
            action_set = set(real_pivots) | set(candidates)
            remaining = [i for i in range(self.n) if i not in action_set]
            extra = self._sample_without_replacement(
                remaining,
                self.k - len(action_set),
            )
            action_set.update(extra)

        action = tuple(sorted(action_set))

        if len(action) != self.k:
            raise RuntimeError(
                f"MNLTopKUIF selected {len(action)} arms instead of k={self.k}: {action}"
            )

        self._last_action = action
        return action

    def update(self, result: RoundResultProtocol) -> None:
        pulled_arms = tuple(int(i) for i in result.pulled_arms)
        pulled_set = set(pulled_arms)

        ranks = dict(result.feedback.censored_ranks)
        full_items = pulled_set | {self.no_choice}

        if len(pulled_set) != self.k:
            raise ValueError(f"Expected exactly k={self.k} real pulled arms.")
        if set(ranks.keys()) != full_items:
            raise ValueError(
                "MNL feedback must contain ranks for pulled arms plus no-choice arm n."
            )

        self.t += 1

        old_accepted = set(self.accepted)
        old_unknown = set(self.unknown)
        old_pivot = int(self.pivot)

        pivot_set = old_accepted | {old_pivot}

        self._rb_update(
            pivot_set=pivot_set,
            items=full_items,
            censored_ranks=ranks,
        )

        A_count, B_count = self._count_ba(
            pivot_set=pivot_set,
            unknown_arms=sorted(old_unknown),
        )

        reject_threshold = self.k - len(old_accepted)
        accept_threshold = len(old_accepted | old_unknown) - self.k

        reject_candidates: set[int] = set()
        accept_candidates: set[int] = set()

        for i in old_unknown:
            if B_count[i] >= reject_threshold:
                reject_candidates.add(i)
            elif A_count[i] >= accept_threshold:
                accept_candidates.add(i)

        new_accepted = set(old_accepted)
        new_unknown = set(old_unknown)

        if self.soft_guards:
            max_rejections = max(0, len(new_accepted | new_unknown) - self.k)
            if len(reject_candidates) > max_rejections:
                reject_candidates = set(
                    self.rng.choice(
                        sorted(reject_candidates),
                        size=max_rejections,
                        replace=False,
                    )
                )

        new_unknown -= reject_candidates

        accept_candidates &= new_unknown

        if self.soft_guards:
            available_slots = max(0, self.k - len(new_accepted))
            if len(accept_candidates) > available_slots:
                accept_candidates = set(
                    self.rng.choice(
                        sorted(accept_candidates),
                        size=available_slots,
                        replace=False,
                    )
                )

        new_accepted |= accept_candidates
        new_unknown -= accept_candidates

        new_pivot = old_pivot

        # Branch condition uses old A_t.
        if len(old_accepted) != 0:
            pivot_accept_candidates = {
                i
                for i in old_unknown
                if i in new_unknown and self.lcb(old_pivot, i) > 0.5
            }

            if self.soft_guards:
                available_slots = max(0, self.k - len(new_accepted))
                if len(pivot_accept_candidates) > available_slots:
                    pivot_accept_candidates = set(
                        self.rng.choice(
                            sorted(pivot_accept_candidates),
                            size=available_slots,
                            replace=False,
                        )
                    )

            new_accepted |= pivot_accept_candidates
            new_unknown -= pivot_accept_candidates

        else:
            candidate = self._new_pivot(
                pivot=old_pivot,
                unknown_arms=sorted(new_unknown),
            )
            if candidate is not None:
                new_pivot = candidate

        self.accepted = new_accepted
        self.unknown = new_unknown
        self.pivot = int(new_pivot)
        self._completed = len(self.accepted) == self.k and len(self.unknown) == 0

    def ucb(self, pivot: int, arm: int) -> float:
        return float(self.mu_hat[pivot, arm] + self.beta_fn(self.N[pivot, arm]))

    def lcb(self, pivot: int, arm: int) -> float:
        return float(self.mu_hat[pivot, arm] - self.beta_fn(self.N[pivot, arm]))

    def state(self) -> AlgorithmState:
        return AlgorithmState(
            t=self.t,
            accepted=tuple(sorted(self.accepted)),
            unknown=tuple(sorted(self.unknown)),
            pivot=int(self.pivot),
            completed=bool(self._completed),
        )

    def diagnostics(self) -> dict:
        return {
            "num_invariant_violations": len(self.invariant_violations),
            "last_invariant_violations": self.invariant_violations[-10:],
        }

    def diagnostic_tables(self) -> dict[str, np.ndarray]:
        return {
            "mu_hat": self.mu_hat.copy(),
            "N": self.N.copy(),
        }

    def _rb_update(
        self,
        *,
        pivot_set: Iterable[int],
        items: Iterable[int],
        censored_ranks: dict[int, int],
    ) -> None:
        for p in pivot_set:
            p = int(p)
            rank_p = censored_ranks[p]

            for i in items:
                i = int(i)
                rank_i = censored_ranks[i]

                if rank_p != rank_i:
                    old_N = self.N[p, i]
                    new_N = old_N + 1.0

                    indicator = 1.0 if rank_i < rank_p else 0.0

                    self.mu_hat[p, i] = (
                        ((new_N - 1.0) / new_N) * self.mu_hat[p, i]
                        + indicator / new_N
                    )
                    self.N[p, i] = new_N

    def _count_ba(
        self,
        *,
        pivot_set: Iterable[int],
        unknown_arms: Iterable[int],
    ) -> tuple[dict[int, int], dict[int, int]]:
        pivots = tuple(int(p) for p in pivot_set)
        unknown = tuple(int(i) for i in unknown_arms)

        A_count = {i: 0 for i in unknown}
        B_count = {i: 0 for i in unknown}

        for i in unknown:
            for j in unknown:
                if any(self.ucb(p, i) < self.lcb(p, j) for p in pivots):
                    B_count[i] += 1

                if any(self.lcb(p, i) > self.ucb(p, j) for p in pivots):
                    A_count[i] += 1

        return A_count, B_count

    def _new_pivot(
        self,
        *,
        pivot: int,
        unknown_arms: Iterable[int],
    ) -> Optional[int]:
        candidates = [
            int(i)
            for i in unknown_arms
            if self.lcb(int(pivot), int(i)) > 0.5
        ]

        if not candidates:
            return None

        scores = np.asarray([self.lcb(pivot, i) for i in candidates], dtype=float)
        max_score = np.max(scores)

        best = [
            arm
            for arm, score in zip(candidates, scores)
            if np.isclose(score, max_score)
        ]

        return int(self.rng.choice(best))

    def _sample_without_replacement(
        self,
        candidates: list[int],
        size: int,
    ) -> tuple[int, ...]:
        if size < 0:
            raise ValueError("Cannot sample a negative number of arms.")
        if size == 0:
            return tuple()
        if len(candidates) < size:
            raise ValueError(
                f"Cannot sample {size} items from only {len(candidates)} candidates."
            )

        sampled = self.rng.choice(candidates, size=size, replace=False)
        return tuple(int(i) for i in sampled)


class SahaGaillardMNLTopK(BaseAlgorithm):
    """
    Saha-Gaillard style AOA-RBPL baseline for MNL bandits with top-k objective.

    Real arms are 0, ..., n - 1.
    The no-choice item is represented internally as arm n.

    adaptive=False:
        AOA-RBPL with no-choice pivot:
            theta_i^ucb = gamma_i0^ucb

    adaptive=True:
        AOA-RBPL-Adaptive:
            theta_i^ucb = min_j gamma_ij^ucb gamma_j0^ucb
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        T: int,
        x: Optional[float] = None,
        x_scale: float = 2.0,
        adaptive: bool = True,
        seed: Optional[int] = None,
        eps: float = 1e-12,
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2.")
        if not (1 <= k <= n):
            raise ValueError("Require 1 <= k <= n.")
        if T < 1:
            raise ValueError("T must be at least 1.")

        self.n = int(n)
        self.k = int(k)
        self.T = int(T)

        self.no_choice = self.n
        self.dim = self.n + 1

        self.adaptive = bool(adaptive)
        self.eps = float(eps)
        self.x = float(x) if x is not None else float(x_scale * np.log(max(T, 2)))

        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.W = np.zeros((self.dim, self.dim), dtype=float)

        self._last_action: Optional[tuple[int, ...]] = None
        self._last_theta_ucb = np.full(self.n, np.inf, dtype=float)

    def select_action(self) -> tuple[int, ...]:
        theta_ucb = self._theta_ucb()
        self._last_theta_ucb = theta_ucb.copy()

        # Random tie-breaking that works even when theta_ucb is huge.
        perm = self.rng.permutation(self.n)
        order_in_perm = np.argsort(-theta_ucb[perm], kind="stable")
        order = perm[order_in_perm]

        action = tuple(sorted(int(i) for i in order[: self.k]))

        self._last_action = action
        return action

    def update(self, result: RoundResultProtocol) -> None:
        pulled = tuple(int(i) for i in result.pulled_arms)
        ranks = dict(result.feedback.censored_ranks)

        outcome_set = set(pulled) | {self.no_choice}

        if len(set(pulled)) != self.k:
            raise ValueError(f"Expected exactly k={self.k} real pulled arms.")
        if set(ranks.keys()) != outcome_set:
            raise ValueError(
                "MNL feedback must contain ranks for pulled arms plus no-choice arm n."
            )

        winners = [arm for arm, rank in ranks.items() if rank == 1]
        if len(winners) != 1:
            raise ValueError(f"Expected exactly one winner, got {winners}.")

        winner = int(winners[0])

        # Rank-breaking winner feedback:
        # the selected item beats all other items in S union {no-choice}.
        for loser in outcome_set:
            loser = int(loser)
            if loser != winner:
                self.W[winner, loser] += 1.0

        self.t += 1

    def state(self) -> AlgorithmState:
        if self._last_action is None:
            action = self.select_action()
        else:
            action = self._last_action

        return AlgorithmState(
            t=self.t,
            accepted=tuple(sorted(action)),
            unknown=tuple(sorted(set(range(self.n)) - set(action))),
            pivot=-1,
            completed=False,
        )

    def diagnostic_tables(self) -> dict[str, np.ndarray]:
        return {
            "W": self.W.copy(),
            "N": (self.W + self.W.T).copy(),
            "theta_ucb": self._last_theta_ucb.copy(),
        }

    def _p_hat_and_ucb(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.W + self.W.T

        p_hat = np.full((self.dim, self.dim), 0.5, dtype=float)
        mask = N > 0
        p_hat[mask] = self.W[mask] / N[mask]
        np.fill_diagonal(p_hat, 0.5)

        p_ucb = np.empty((self.dim, self.dim), dtype=float)

        # Unobserved pairs are maximally optimistic.
        p_ucb[~mask] = 1.0 - self.eps

        ph = p_hat[mask]
        Nm = N[mask]
        bonus = np.sqrt(2.0 * ph * (1.0 - ph) * self.x / Nm) + 3.0 * self.x / Nm
        p_ucb[mask] = ph + bonus

        p_ucb = np.clip(p_ucb, self.eps, 1.0 - self.eps)
        np.fill_diagonal(p_ucb, 0.5)

        return p_hat, p_ucb

    def _gamma_ucb(self) -> np.ndarray:
        _, p_ucb = self._p_hat_and_ucb()

        gamma = p_ucb / np.maximum(1.0 - p_ucb, self.eps)
        np.fill_diagonal(gamma, 1.0)
        return gamma

    def _theta_ucb(self) -> np.ndarray:
        gamma = self._gamma_ucb()

        if not self.adaptive:
            return gamma[: self.n, self.no_choice].copy()

        gamma_to_no_choice = gamma[:, self.no_choice]
        theta_ucb = np.empty(self.n, dtype=float)

        for i in range(self.n):
            theta_ucb[i] = float(np.min(gamma[i, :] * gamma_to_no_choice))

        return theta_ucb