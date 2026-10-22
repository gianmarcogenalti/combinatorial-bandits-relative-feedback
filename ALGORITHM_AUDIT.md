# Implementation notes

## UIF and MNL-UIF

`experiment_code/algorithms.py` implements Top-k-UIF and its MNL adaptation.
Each round uses snapshots of the accepted set, unknown set, and pivot from the
start of that round. The played assortment contains accepted real arms and the
real pivot, with remaining positions sampled uniformly from unknown arms.
The fictitious MNL no-choice item is represented internally by index n and does
not occupy a real assortment position.

Rank breaking updates a pair only when its observed ranks differ. Thus two
censored items produce no comparison. The Count-BA relation is closed
transitively, including chains whose comparisons use different pivots. Pivot
selection uses an exact maximum lower confidence bound with uniform tie breaking.
After the first MNL acceptance, an accepted real arm replaces the no-choice pivot
if the ordinary pivot rule returns no candidate.

Acceptance and rejection use the pre-round thresholds. Cardinality guards are
applied in ascending arm order and their activations are checkpointed. Acceptance
requires fewer than k accepted arms; rejection requires more than k remaining
accepted-or-unknown arms. Once k arms are accepted, the unknown set is cleared
and that assortment is played for the remaining horizon. The additional
pivot-based acceptance rule examines the pre-round unknown set.

For positive observation count N, both UIF variants use:

```text
beta(N) = sqrt(log(2*n*T/delta) / N),   delta = 0.05.
```

There is no multiplicative rescaling. Set beta(0)=infinity and beta(infinity)=0
for diagonal self-comparisons. The horizon T is fixed in each configuration.
The choice delta=0.05 is an experimental setting; the paper's horizon-dependent
regret guarantee uses delta=1/T.

With one pivot, the strict interval relation is already transitive, so sorted
endpoints give the same comparison counts without a dense transitive closure.
With several pivots, the full cross-pivot closure is retained.

## Baselines

**RecMaxMinUCB.** `experiment_code/recmaxmin.py` contains the recursive baseline
extracted without algorithm changes from the original implementation at revision
`dbd226e`. The EWS adapter in `experiment_code/algorithms.py` supplies injected
randomness, checkpoint state, and exact maximum-score tie breaking. It uses full
ranking feedback, k=m=2, and alpha=0.51. The subset-construction routine is a
cardinality-preserving interpretation of the published pseudocode: it selects
Condorcet candidates individually, recomputes after each selection, then uses
max-min completion. The original printed routine contains ambiguous loop and
set updates, so this implementation is not claimed to be a literal transcription.
Source: [Saha and Gopalan, 2019](https://arxiv.org/pdf/1903.00543), Algorithms 2–3.

The adapter evaluates queried UCB entries lazily and caches the largest k
Condorcet thresholds per row. Selection removes at most k-1 arms from the full
pool, so at least one largest eligible threshold remains cached. Only played
rows need refreshing. Decisions use the original UCB expression, with literal
row checks near floating-point boundaries; caches are rebuilt from checkpoint
counts. Regression tests compare complete actions, counts, ordered slots, RNG
state, and checkpoint continuation against the dense implementation.

**AOA-RB-PL and AOA-RB-PL-Adaptive.** Pairwise probability bounds use the original
empirical-Bernstein radius:

```text
sqrt(2*p_hat*(1-p_hat)*x/N) + 3*x/N,   x = 2*log(T).
```

Unobserved or saturated probability bounds give infinite parameter-ratio bounds;
diagonal ratios are one. The direct method uses ratios to no-choice, while the
adaptive method minimizes the product through all intermediate pivots. Infinite
score ties are randomized. Ratios are not clipped to arbitrary finite constants.
Source: [Saha and Gaillard, 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/b9a4d7b88a41652c63962ebcc21701b7-Paper-Conference.pdf),
Equation (3), Algorithm 1, and Section 4.

## Sampling, measurement, and reproducibility

An exponential race samples the PL ranking law. MNL feedback selects the winner
among the real assortment and no-choice item. All configured preference vectors
are positive, normalized, and have a unique top-k set; internal ties are allowed.
The three top-k boundary gaps are 0.95, 0.10, and 0.15, respectively.

The learner receives rankings only. Evaluation computes regret from the saved
instance and every action, retains 200 exact cumulative-regret points per run, then aggregates independent repetitions using mean
and SEM. EWS owns random-stream persistence; learner checkpoints contain all
other mutable decision state. `tests/test_ews.py` checks checkpoint continuation,
rank breaking, transitive comparisons, pivot handling, and infinite optimism.

The current configuration uses root seed 104729, 20 independent repetitions per
setting, and T=1,000,000. Outputs from different source revisions or configurations
have distinct scientific identities and must not be relabeled as this study.
