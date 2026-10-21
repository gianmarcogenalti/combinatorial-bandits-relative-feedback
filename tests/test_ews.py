"""Scientific regressions and checkpoint replay for the EWS migration."""

import copy

import numpy as np
import pytest

from experiment_code.algorithms import AOARB, UIF, RecMaxMinUCB
from experiment_code.data import RankingBandit


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
def test_uif_uses_unscaled_manuscript_confidence_radius(mnl):
    """Both algorithms use n real arms in log(2*n*T/delta), with unit prefactor."""
    algorithm = UIF(
        rng=np.random.default_rng(3), n=10, k=3, mnl=mnl, horizon=10_000_000, delta=0.05
    )
    np.testing.assert_allclose(algorithm.log_bonus, np.log(2 * 10 * 10_000_000 / 0.05))
    # Increasing the comparison count by four must divide the radius by two.
    radii = np.sqrt(algorithm.log_bonus / np.array([100, 400]))
    assert radii[0] == 2 * radii[1]
    with pytest.raises(TypeError):
        UIF(rng=np.random.default_rng(3), n=10, k=3, beta_scale=0.25)


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
