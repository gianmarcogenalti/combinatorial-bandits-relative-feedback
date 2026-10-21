# Algorithm audit for the EWS rerun

Audited 2026-10-21 against the working manuscript, including its pre-existing
uncommitted algorithm/proof edits. Original experiment revision: `dbd226e`.
The paper and experiment repositories remain separate; old results are preserved.

| Paper statement / source | EWS implementation | Decision / difference from legacy |
| --- | --- | --- |
| Algorithms 1/3, `algs/top_k_uif.tex`, `algs/mnl_top_k_uif.tex` | `experiment_code/algorithms.py:UIF` | Common implementation, pre-round A/U/P, uniform exploration; no-choice has internal index n rather than paper index 0. |
| RB-Update, `algs/subroutines.tex` | `UIF.observe` | Only different observed ranks update; censored/censored pairs do not. Pivot rows store wins of the column arm over that pivot. |
| Count-BA transitive closure | Boolean Floyd-Warshall | Legacy already implements closure; keep it, including paths witnessed by different pivots. |
| New-Pivot exact argmax over old U | `UIF.observe` | Legacy `algorithms.py:504,1346` uses `isclose`, permitting inferior near-maxima. New port uses exact maximizers. |
| First accepted MNL arm replaces no-choice if New-Pivot has no candidate, Algorithm 3 | `UIF.observe` | Missing in legacy `1238–1246`. Substantive: at theta0=1 the good-event LCBs cannot exceed 1/2, yet Count-BA can accept real arms. |
| Clear U when A has k arms, Algorithms 1/3 | `UIF.observe` | Missing in legacy `381–386,1248–1251`; fixes completion diagnostics and redundant updates. |
| Pivot-based acceptance loops over old U | `UIF.observe` | Legacy `354,1221` restricts to remaining U. This only differs outside the confidence event. |
| Cardinality guards, `appendix/C_alg_details.tex:11` | `UIF.observe` | Guard each acceptance/rejection in ascending arm order; count activations. Legacy caps candidate batches using random selection. Equivalent on the confidence event. |
| Confidence bounds | explicit `beta_scale=.25`, delta=.05, horizon=1e6 | Preserve practical legacy radius, not theorem coefficient 1 and delta=1/T. MNL uses n+1 in its log as before. These radii do not automatically inherit the theorem guarantees. |
| Three numerical instances, `content/06_num_sim.tex`, `appendix/F_num_sim.tex` | `experiment.yml` | Same fixed theta values, k and T; use 30 trials for sensitivity (legacy config says 10); explicitly choose m=1,2,3,6. |
| Regret is difference of average selected preferences, also for MNL | `metrics.py:TopKRegret` | Compute from every saved action, never reconstruct cumulative regret from sparse actions. |
| Captions: mean +/- SEM | repeated-runs standard_error; `figures.py` | Legacy PDF exporters use 1.96 SEM. New bands and error bars use plain SEM. |

The original n=10 configuration shuffles arm labels with seed 123. The port uses
the equivalent ordered fixed instance stated in the paper. EWS stream identities
also differ from old seed offsets; numerical reruns are independent, not identical
old trajectories. Exponential races sample exactly the PL distribution, changing
the sampling implementation from sequential choices without changing its law.

## Baselines

**Saha–Gaillard.** `AOARB` uses the original empirical-Bernstein radius
sqrt(2 p-hat (1-p-hat) x/N) + 3x/N, with x=2 log(T), and either direct ratios to
no-choice or the minimum product through all intermediate pivots. The legacy
`algorithms.py:1497,1504,1512` clips upper probabilities to 1-1e-12, turning
infinite optimism into large finite ratios. Products then distinguish arms that
should tie at infinity, changing exploration. The new implementation uses exact
infinity for unobserved/saturated bounds, diagonal ratios one, and random exact
ties. This intentionally corrects the baseline and may change its curve.
Source: [ICLR 2025 original](https://proceedings.iclr.cc/paper_files/paper/2025/file/b9a4d7b88a41652c63962ebcc21701b7-Paper-Conference.pdf),
Equation (3), Algorithm 1, Section 4 (printed pages 6 and 9).

**RecMaxMinUCB.** Reuse the original repository's recursive construction through
a small EWS subclass; save W, time and ordered B slots. Use full ranking k=m=3
and alpha=.51. Replace approximate score ties (`algorithms.py:984`) with exact
maximizers. The inherited build-S routine is a cardinality-preserving
interpretation, not a verified literal transcription: the original printed
routine has ambiguous loop/set updates, while legacy selects Condorcet candidates
individually and recomputes after each. Source:
[Saha–Gopalan arXiv v2](https://arxiv.org/pdf/1903.00543), Definition 1(ii),
Algorithm 2 (printed page 21), Algorithm 3 (printed page 40).

All instances have positive normalized preferences and a unique top-k set;
internal ties in the n=10 instance are allowed. Boundary gaps are .10, .10 and
.15 respectively. No redesign of the proposed algorithms or theorem is implied.

## Verification

`tests/test_ews.py` checks deterministic continuation for all five learner modes,
the no-choice fallback, terminal clearing, censored-pair rank breaking, PL winner
frequencies, and infinite baseline ties. The 36-run smoke configuration exercises
EWS instance persistence, checkpoints, complete-action regret, aggregation,
retention and all three TikZ outputs. See the sibling paper task
`tasks/014_ews_tikz_experiments/task.md` for execution results and the cloud run ID.
