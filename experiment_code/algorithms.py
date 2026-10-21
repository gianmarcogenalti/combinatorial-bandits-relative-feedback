"""Paper algorithms with injected random streams and explicit checkpoint state."""

import numpy as np

from .recmaxmin import RecMaxMinUCB as LegacyRecMaxMinUCB


class UIF:
    """Algorithms 1/3 and their shared subroutines, using pre-round A, U and P.

    Rows of counts/wins correspond to pivots. No-choice is indexed by n.
    Confidence radii are sqrt(log(2*n*T/delta)/N), without rescaling. Guards follow
    Appendix C, using ascending arm order; activations are counted in checkpoints.
    """

    supports_extension = False  # Confidence bounds depend on the fixed horizon.

    def __init__(self, *, rng, n, k, m=1, mnl=False, horizon=1000000, delta=0.05):
        if not (2 <= k <= n and 1 <= m <= k and 0 < delta <= 1 and horizon >= 1):
            raise ValueError("Invalid UIF parameters")
        self.rng, self.n, self.k, self.m, self.mnl = rng, n, k, m, mnl
        self.log_bonus = np.log(2 * n * horizon / delta)
        dim = n + int(mnl)
        self.counts = np.zeros((dim, dim))
        self.wins = np.zeros((dim, dim))
        np.fill_diagonal(self.counts, np.inf)
        self.accepted = np.zeros(n, dtype=bool)
        self.unknown = np.ones(n, dtype=bool)
        self.pivot = n if mnl else int(rng.integers(n))
        self.round = 0
        self.completed_round = 0
        self.guard_events = 0

    def act(self, context=None):
        accepted = np.flatnonzero(self.accepted)
        if len(accepted) == self.k:
            return accepted
        pivots = np.union1d(accepted, [self.pivot])
        real_pivots = pivots[pivots < self.n]
        candidates = np.setdiff1d(np.flatnonzero(self.unknown), [self.pivot])
        size = self.k - len(real_pivots)
        if not 0 <= size <= len(candidates):
            raise RuntimeError("UIF invariant failed: cannot sample a size-k action")
        sampled = self.rng.choice(candidates, size=size, replace=False)
        return np.sort(np.concatenate((real_pivots, sampled)))

    def observe(self, action, feedback):
        self.round += 1
        if self.completed_round:
            return
        old_a, old_u, old_p = self.accepted.copy(), self.unknown.copy(), self.pivot
        pivots = np.union1d(np.flatnonzero(old_a), [old_p])
        items = np.append(action, self.n) if self.mnl else np.asarray(action)
        ranks = np.asarray(feedback)
        pi = np.ix_(pivots, items)
        comparable = ranks[pivots, None] != ranks[items]
        self.counts[pi] += comparable
        self.wins[pi] += (ranks[items] < ranks[pivots, None]) & comparable

        unknown = np.flatnonzero(old_u)
        counts = self.counts[np.ix_(pivots, unknown)]
        means = np.full(counts.shape, 0.5)
        observed = (counts > 0) & np.isfinite(counts)
        np.divide(self.wins[np.ix_(pivots, unknown)], counts, out=means, where=observed)
        with np.errstate(divide="ignore"):
            bonus = np.sqrt(self.log_bonus / counts)
        lower, upper = means - bonus, means + bonus
        precedes = np.any(lower[:, :, None] > upper[:, None, :], axis=0)
        # Boolean Floyd-Warshall computes the Count-BA transitive closure.
        for j in range(len(unknown)):
            precedes |= precedes[:, j, None] & precedes[j, None, :]
        np.fill_diagonal(precedes, False)
        reject = precedes.sum(axis=0) >= self.k - old_a.sum()
        accept = (~reject) & (precedes.sum(axis=1) >= (old_a | old_u).sum() - self.k)
        # The displayed pseudocode omits the Appendix C cardinality guards.
        for arm, rejected, accepted in zip(unknown, reject, accept):
            if rejected:
                if (self.accepted | self.unknown).sum() > self.k:
                    self.unknown[arm] = False
                else:
                    self.guard_events += 1
            elif accepted:
                if self.accepted.sum() < self.k:
                    self.accepted[arm], self.unknown[arm] = True, False
                else:
                    self.guard_events += 1

        pivot_lower = lower[np.flatnonzero(pivots == old_p)[0]]
        better = pivot_lower > 0.5
        if old_a.any():
            # All of U_t participates, including arms just removed from U.
            for arm in unknown[better]:
                if self.accepted[arm]:
                    continue
                if self.accepted.sum() < self.k:
                    self.accepted[arm], self.unknown[arm] = True, False
                else:
                    self.guard_events += 1
        elif better.any():
            best = better & (pivot_lower == np.max(pivot_lower[better]))
            self.pivot = int(self.rng.choice(unknown[best]))
        elif self.mnl and old_p == self.n and self.accepted.any():
            self.pivot = int(self.rng.choice(np.flatnonzero(self.accepted)))

        if self.accepted.sum() > self.k or np.any(self.accepted & self.unknown):
            raise RuntimeError("UIF accept/reject invariant failed")
        if self.accepted.sum() == self.k:
            self.unknown[:] = False
            self.completed_round = self.round

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in (
                "counts",
                "wins",
                "accepted",
                "unknown",
                "pivot",
                "round",
                "completed_round",
                "guard_events",
            )
        }

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value.copy() if isinstance(value, np.ndarray) else value)


class RecMaxMinUCB(LegacyRecMaxMinUCB):
    """Reuse the original recursive baseline through the EWS online interface.

    The inherited module is fingerprinted by EWS. Checkpoint ordered B slots
    explicitly and use exact maximizers when breaking ties.
    """

    supports_extension = True

    def __init__(self, *, rng, n, k, m, alpha=0.51):
        super().__init__(n=n, k=k, m=m, alpha=alpha, seed=0)
        self.rng = rng

    def act(self, context=None):
        return np.asarray(self.select_action(), dtype=np.int64)

    def observe(self, action, feedback):
        ranks = np.asarray(feedback)[action]
        self.W[np.ix_(action, action)] += ranks[:, None] < ranks[None, :]
        self.t += 1

    def _choose_one(self, candidates, *, U, selected):
        choices = np.asarray(sorted(candidates))
        if selected:
            scores = np.min(U[np.ix_(choices, sorted(selected))], axis=1)
            choices = choices[scores == scores.max()]
        return int(self.rng.choice(choices))

    def state_dict(self):
        return {"W": self.W, "t": self.t, "B_slots": [sorted(s) for s in self.B_slots]}

    def load_state_dict(self, state):
        self.W = np.array(state["W"], copy=True)
        self.t = int(state["t"])
        self.B_slots = [set(s) for s in state["B_slots"]]
        self._last_action = None


class AOARB:
    """Saha-Gaillard pairwise-ratio optimism with optional intermediate pivots.

    Saturated probability bounds yield infinite ratios as in the paper, rather
    than arbitrary large finite proxies. The no-choice ratio to itself is one.
    """

    supports_extension = False

    def __init__(self, *, rng, n, k, horizon=1000000, adaptive=True, x_scale=2.0):
        self.rng, self.n, self.k, self.adaptive = rng, n, k, adaptive
        self.x = x_scale * np.log(max(horizon, 2))
        self.wins = np.zeros((n + 1, n + 1))

    def act(self, context=None):
        counts = self.wins + self.wins.T
        mean = np.divide(
            self.wins, counts, out=np.full(counts.shape, 0.5), where=counts > 0
        )
        safe_counts = np.maximum(counts, 1)
        upper = (
            mean
            + np.sqrt(2 * mean * (1 - mean) * self.x / safe_counts)
            + 3 * self.x / safe_counts
        )
        upper[counts == 0] = 1.0
        ratios = np.divide(
            upper, 1 - upper, out=np.full(upper.shape, np.inf), where=upper < 1
        )
        np.fill_diagonal(ratios, 1.0)
        scores = ratios[: self.n, self.n]
        if self.adaptive:
            scores = np.min(ratios[: self.n] * ratios[:, self.n], axis=1)
        permutation = self.rng.permutation(self.n)
        order = np.argsort(-scores[permutation], kind="stable")
        return np.sort(permutation[order[: self.k]])

    def observe(self, action, feedback):
        items = np.append(action, self.n)
        winner = items[np.argmin(np.asarray(feedback)[items])]
        self.wins[winner, items[items != winner]] += 1

    def state_dict(self):
        return {"wins": self.wins}

    def load_state_dict(self, state):
        self.wins = np.array(state["wins"], copy=True)
