"""Scientific regression and checkpoint replay tests."""

import copy
import itertools

import numpy as np
import pytest

from experiment_code.algorithms import AOARB, UIF, RecMaxMinUCB
from experiment_code.data import RankingBandit
from experiment_code.recmaxmin import RecMaxMinUCB as LegacyRecMaxMinUCB


def make_data(n=10, k=3, m=3, mnl=False):
    params = dict(theta=np.linspace(1, 0.2, n).tolist(), k=k, m=m, mnl=mnl)
    instance = RankingBandit.create_instance(rng=np.random.default_rng(1), **params)
    return RankingBandit(rng=np.random.default_rng(2), instance=instance, **params)


@pytest.mark.parametrize(
    "cls,params",
    [
        (UIF, dict(n=10, k=3, m=3)),
        (UIF, dict(n=10, k=3, mnl=True)),
        (RecMaxMinUCB, dict(n=10, k=3, m=3)),
        (AOARB, dict(n=10, k=3, adaptive=True)),
        (AOARB, dict(n=10, k=3, adaptive=False)),
    ],
)
def test_checkpoint_replays_future_actions_and_feedback(cls, params):
    """Restoring learner state and independent RNG streams reproduces the suffix."""
    mnl = params.get("mnl", cls is AOARB)
    data = make_data(m=1 if mnl else 3, mnl=mnl)
    algorithm = cls(rng=np.random.default_rng(3), **params)

    def advance():
        action = algorithm.act()
        feedback = data.generate(action).value
        algorithm.observe(action, feedback)
        return action.copy(), feedback.copy()

    for _ in range(80):
        advance()
    saved = copy.deepcopy(algorithm.state_dict())
    alg_rng = copy.deepcopy(algorithm.rng.bit_generator.state)
    data_rng = copy.deepcopy(data.rng.bit_generator.state)
    expected = [advance() for _ in range(120)]
    algorithm = cls(rng=np.random.default_rng(999), **params)
    algorithm.load_state_dict(saved)
    algorithm.rng.bit_generator.state = alg_rng
    data.rng.bit_generator.state = data_rng
    for action, feedback in expected:
        actual_action, actual_feedback = advance()
        np.testing.assert_array_equal(action, actual_action)
        np.testing.assert_array_equal(feedback, actual_feedback)


def test_mnl_first_acceptance_replaces_no_choice_pivot():
    """Count-BA can accept a real arm even when no arm beats no-choice."""
    algorithm = UIF(rng=np.random.default_rng(3), n=4, k=2, mnl=True)
    algorithm.counts[4, :4] = 100000
    algorithm.wins[4, :4] = [40000, 20000, 20000, 20000]
    algorithm.observe(np.array([0, 1]), np.array([2, 2, 2, 2, 1]))
    assert algorithm.accepted.tolist() == [True, False, False, False]
    assert algorithm.pivot == 0


def test_terminal_cleanup_after_pivot_shortcut():
    """Accepting the kth arm clears every remaining unknown arm immediately."""
    algorithm = UIF(rng=np.random.default_rng(3), n=4, k=2)
    algorithm.pivot = 0
    algorithm.accepted[0] = True
    algorithm.unknown[0] = False
    algorithm.counts[0, 1] = 100000
    algorithm.wins[0, 1] = 80000
    algorithm.observe(np.array([0, 1]), np.array([2, 1, 2, 2]))
    assert algorithm.accepted.sum() == 2
    assert not algorithm.unknown.any()
    assert algorithm.completed_round == 1
    np.testing.assert_array_equal(algorithm.act(), [0, 1])


def test_rank_breaking_keeps_censored_pairs_unobserved():
    algorithm = UIF(rng=np.random.default_rng(3), n=4, k=3)
    algorithm.pivot = 1
    algorithm.observe(np.array([0, 1, 2]), np.array([1, 2, 2, 2]))
    assert algorithm.counts[1, 0] == 1
    assert algorithm.wins[1, 0] == 1
    assert algorithm.counts[1, 2] == 0


def test_transitive_comparisons_can_use_different_pivots():
    """A chain through two pivots certifies an acceptance no direct edge can."""
    algorithm = UIF(rng=np.random.default_rng(3), n=5, k=3)
    algorithm.pivot = 0
    algorithm.accepted[:2] = True
    algorithm.unknown[:2] = False
    # Pivot 0 witnesses 2 > 3; pivot 1 witnesses 3 > 4. Other pairs unobserved.
    algorithm.counts[0, [2, 3]] = 100000
    algorithm.wins[0, [2, 3]] = [40000, 20000]
    algorithm.counts[1, [3, 4]] = 100000
    algorithm.wins[1, [3, 4]] = [40000, 20000]
    algorithm.observe(np.array([0, 1, 2]), np.array([2, 2, 1, 2, 2]))
    assert algorithm.accepted.tolist() == [True, True, True, False, False]
    assert algorithm.completed_round == 1


def test_pl_winner_frequencies_match_preferences():
    data = make_data(n=3, k=3, m=1)
    counts = np.zeros(3)
    for _ in range(20000):
        counts[np.argmin(data.generate([0, 1, 2]).value)] += 1
    np.testing.assert_allclose(
        counts / counts.sum(), data.theta / data.theta.sum(), atol=0.012
    )


def test_adaptive_unknown_ratio_bounds_remain_infinite_ties():
    """Unsaturated intermediate ratios cannot turn an unknown bound finite."""
    algorithm = AOARB(rng=np.random.default_rng(3), n=3, k=2)
    algorithm.wins[0, 1] = 100
    algorithm.wins[1, 0] = 1000
    # All arms still have infinite bounds to no-choice; across seeds all are sampled.
    selections = {tuple(algorithm.act()) for _ in range(100)}
    assert selections == {(0, 1), (0, 2), (1, 2)}


@pytest.mark.parametrize("mnl", [False, True])
def test_uif_uses_manuscript_confidence_radius(mnl):
    """Both algorithms use n real arms in log(2*n*T/delta)."""
    algorithm = UIF(
        rng=np.random.default_rng(3), n=10, k=3, mnl=mnl, horizon=10_000_000, delta=0.05
    )
    np.testing.assert_allclose(algorithm.log_bonus, np.log(2 * 10 * 10_000_000 / 0.05))
    # Increasing the comparison count by four must divide the radius by two.
    radii = np.sqrt(algorithm.log_bonus / np.array([100, 400]))
    assert radii[0] == 2 * radii[1]


def test_regret_downsampling_includes_every_action():
    """An unretained intermediate action must still contribute to final regret."""
    from types import SimpleNamespace

    from experiment_code.metrics import TopKRegret

    class Saved(dict):
        instance = SimpleNamespace(arrays={"theta": np.array([1.0, 0.5, 0.1])})

    result = Saved(
        step=np.arange(1, 6), action=np.array([[0, 1], [1, 2], [0, 1], [1, 2], [0, 1]])
    )
    curve = TopKRegret(points=2).compute(result)
    np.testing.assert_array_equal(curve.x, [1, 5])
    np.testing.assert_allclose(curve.values, [0.0, 0.9])


@pytest.mark.parametrize("n", [1, 10, 50])
def test_single_pivot_interval_counts_equal_general_closure(n):
    from experiment_code.algorithms import _comparison_counts

    rng = np.random.default_rng(5)
    for _ in range(30):
        lo = rng.normal(size=n)
        hi = lo + rng.exponential(size=n)
        lo[0] = hi[0] = 0.5  # The pivot has an exact self-comparison.
        if n > 1:
            lo[1], hi[1] = -np.inf, np.inf
        edges = lo[:, None] > hi[None, :]
        for j in range(n):
            edges |= edges[:, j, None] & edges[j, None, :]
        actual_a, actual_b = _comparison_counts(lo[None, :], hi[None, :])
        np.testing.assert_array_equal(actual_a, edges.sum(axis=1))
        np.testing.assert_array_equal(actual_b, edges.sum(axis=0))


def test_vectorized_recursive_candidates_match_strict_published_predicate():
    rng = np.random.default_rng(4)
    alg = RecMaxMinUCB(rng=rng, n=20, k=3, m=3)
    for _ in range(50):
        u = rng.choice([0.0, 0.5, 0.75, np.inf], size=(20, 20))
        pool = set(map(int, rng.choice(20, size=10, replace=False)))
        expected = {i for i in pool if all(u[i, j] > 0.5 for j in pool if i != j)}
        assert alg._condorcet_candidates(u, pool) == expected
    assert alg._condorcet_candidates(u, set()) == set()


class DenseRecMaxMinUCB(RecMaxMinUCB):
    """Literal baseline UCBs/predicate as an independent optimization oracle."""

    _ucb_matrix = LegacyRecMaxMinUCB._ucb_matrix
    _condorcet_candidates = LegacyRecMaxMinUCB._condorcet_candidates


@pytest.mark.parametrize("k", [2, 3, 5])
def test_lazy_scores_and_strict_predicates_at_float_boundaries(k):
    rng = np.random.default_rng(118 + k)
    n, t = 12, 17003
    base = DenseRecMaxMinUCB(rng=np.random.default_rng(1), n=n, k=k, m=k)
    for i in range(n):
        for j in range(i):
            N = float(rng.choice([200, 10000, 1e12, 2**52]))
            p = 0.5 - np.sqrt(base.alpha * np.log(t + 1) / N)
            for _ in range(int(rng.integers(0, 8))):
                p = np.nextafter(p, rng.choice([-np.inf, np.inf]))
            base.W[i, j], base.W[j, i] = p * N, (1 - p) * N
    base.t = t
    fast = RecMaxMinUCB(rng=np.random.default_rng(1), n=n, k=k, m=k)
    fast.load_state_dict(base.state_dict())
    literal, lazy = base._ucb_matrix(t + 1), fast._ucb_matrix(t + 1)
    np.testing.assert_array_equal(literal, np.asarray(lazy))
    for omitted_count in range(min(k, 4)):
        for omitted in itertools.combinations(range(n), omitted_count):
            pool = set(range(n)) - set(omitted)
            assert base._condorcet_candidates(
                literal, pool
            ) == fast._condorcet_candidates(lazy, pool)
    for pool in [{0}, {0, 2, 5}, set()]:
        assert base._condorcet_candidates(literal, pool) == fast._condorcet_candidates(
            lazy, pool
        )
    np.testing.assert_array_equal(base._ucb_matrix(1), np.asarray(fast._ucb_matrix(1)))
    pool = set(range(n))
    assert base._condorcet_candidates(
        base._ucb_matrix(1), pool
    ) == fast._condorcet_candidates(fast._ucb_matrix(1), pool)
    assert fast.boundary_rows > 0


@pytest.mark.parametrize("k,seed", [(2, 0), (2, 1), (3, 0), (3, 1)])
def test_actual_pl_10000_rounds_and_checkpoint_continuation(k, seed):
    n = 32
    theta = np.array([1.0] * k + [0.05] * (n - k))
    base = DenseRecMaxMinUCB(rng=np.random.default_rng(seed + 21), n=n, k=k, m=k)
    fast = RecMaxMinUCB(rng=np.random.default_rng(seed + 21), n=n, k=k, m=k)
    env = np.random.default_rng(seed + 31)
    for t in range(10000):
        a, b = base.act(), fast.act()
        np.testing.assert_array_equal(a, b)
        order = np.argsort(env.exponential(1 / theta[a]))
        ranks = np.full(n, k, dtype=int)
        ranks[a[order]] = np.arange(1, k + 1)
        base.observe(a, ranks)
        fast.observe(b, ranks)
        assert base.t == fast.t and base.B_slots == fast.B_slots
        if t % 100 == 0:
            np.testing.assert_array_equal(base.W, fast.W)
        if t == 4000:
            state = copy.deepcopy(fast.state_dict())
            rngstate = copy.deepcopy(fast.rng.bit_generator.state)
            fast = RecMaxMinUCB(rng=np.random.default_rng(999), n=n, k=k, m=k)
            fast.load_state_dict(state)
            fast.rng.bit_generator.state = rngstate
    np.testing.assert_array_equal(base.W, fast.W)
    assert base.rng.bit_generator.state == fast.rng.bit_generator.state


def test_cancellation_can_hide_a_second_comparison_at_huge_counts():
    """A smaller real-valued threshold can still round to the strict .5 tie."""
    n, k = 4, 2
    base = DenseRecMaxMinUCB(rng=np.random.default_rng(1), n=n, k=k, m=k)
    d = 2.0**-26
    N2 = float(3 * 2**54)
    C2 = N2 * d * d / base.alpha
    round_number = int(np.ceil(np.exp(C2 + 2e-9)))
    log_t = np.log(round_number)
    p1, p2 = 0.1, 0.5 - d
    N1 = base.alpha * (log_t - 1e-9) / (0.5 - p1) ** 2
    base.W[0, 1], base.W[1, 0] = p1 * N1, (1 - p1) * N1
    base.W[0, 2], base.W[2, 0] = p2 * N2, (1 - p2) * N2
    base.t = round_number - 1
    fast = RecMaxMinUCB(rng=np.random.default_rng(1), n=n, k=k, m=k)
    fast.load_state_dict(base.state_dict())
    literal, lazy = base._ucb_matrix(round_number), fast._ucb_matrix(round_number)
    assert literal[0, 1] > 0.5 and literal[0, 2] == 0.5
    pool = set(range(n))
    assert base._condorcet_candidates(literal, pool) == fast._condorcet_candidates(
        lazy, pool
    )
    assert fast.boundary_rows > 0
