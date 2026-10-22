# CMABs with ranking feedback: experiments

Reproducible experiments for combinatorial bandits with Plackett–Luce ranking
feedback and multinomial-logit bandits with a top-k objective. The complete
workflow uses [experiments-wo-stress (EWS)](https://github.com/m1gwings/experiments-wo-stress).

## Setup and execution

Use Python 3.10–3.12 and Git. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ews run experiment.yml --output outputs/paper --workers 2
```

`requirements.txt` pins EWS and its numerical dependencies. `ews run` executes
simulations, computes regret, aggregates repetitions, and exports the figures.
The worker count controls parallelism without changing the random streams.
To resume interrupted work, repeat the same command with the same output path.
Inspect saved progress with:

```bash
ews inspect outputs/paper
```

## Experimental design

`experiment.yml` specifies 20 independent repetitions and a horizon of 1,000,000
rounds for each setting and algorithm, totaling 360 runs:

| Study | Arms n | Assortment size k | Feedback | Algorithms |
| --- | ---: | ---: | --- | --- |
| Comparison | 400 | 2 | Full ranking, m=2 | Top-k-UIF; RecMaxMinUCB |
| Feedback sensitivity | 16 | 6 | m in {1,2,3,6} | Top-k-UIF |
| MNL | 30 | 5 | Winner feedback; theta0 in {0.001,0.003,0.01,0.03} | MNL-Top-k-UIF; AOA-RB-PL; AOA-RB-PL-Adaptive |

The comparison uses two arms of weight 1 and 398 arms of weight 0.05.
The MNL comparison studies the small-no-choice regime shown in the table.
The preference vectors and all algorithm parameters are explicit in the
configuration. UIF uses the confidence radius `sqrt(log(2*n*T/delta)/N)` with
`delta=0.05`, infinite radius for unobserved pairs, and zero radius for diagonal
self-comparisons.

The root seed is 104729. EWS supplies separate reproducible NumPy streams for the
learner and environment and checkpoints their states. Regret is the cumulative
difference between the average preference of the optimal k arms and that of
the played assortment, also for MNL. It is computed from every recorded action
before temporal downsampling. Aggregates report the empirical mean and pointwise
standard error of the mean (SEM) over independent repetitions. The figure
exporter uses these aggregates to construct pointwise 95% Student-t confidence
intervals for the mean: `mean ± t(0.975, N-1) * SEM`. With 20 repetitions,
the critical value is approximately 2.093.

## Outputs and figures

EWS stores checkpoints, provenance, instances, and analysis under the selected
output directory. The three native PGFPlots fragments are exported under
`analysis/figures/`:

- `cumulative_regret.tikz`
- `final_regret_vs_m.tikz`
- `final_regret_vs_theta0.tikz`

The fragments inherit the enclosing LaTeX document's fonts. Load `pgfplots`,
load its `fillbetween` library and `fix-cm`, and set `\pgfplotsset{compat=1.18}` before
including a fragment with `\input{PATH/NAME.tikz}`. Bands and error bars are
pointwise 95% confidence intervals for the mean. Saved aggregate curves contain
200 time points, including both endpoints.

To update figure formatting from compatible saved analysis:

```bash
ews plot experiment.yml --output outputs/paper
```

The configured retention policy discards trajectories after their metrics are
saved. Each run retains 200 exact cumulative-regret points, computed using every
action before downsampling, so individual runs and uncertainty can be inspected.
New metrics requiring discarded trajectories require a new run.

## Validation

```bash
python -m pip install pytest
python -m pytest -q tests/
```

## Saved results

The checked-in [reference results](reference_results/) contain the aggregate
curves from all 360 completed trials, final means with SEM and 95% confidence
limits, and the three paper figures. Source/configuration provenance and file checksums are recorded in
`reference_results/provenance.json`. Use the matching source and pinned
dependencies when reproducing those curves.
