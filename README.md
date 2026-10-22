# Experiments for Top-k-UIF and MNL-Top-k-UIF

Cristiano Migali, Gianmarco Genalti, Alberto Maria Metelli, and Marco Mussi

Code for the experiments in *Worst-Case Regret Bounds for Combinatorial Bandits
with Ranking Feedback*.

## How to Run the Experiments

The experiments use [experiments-wo-stress (EWS)](https://github.com/m1gwings/experiments-wo-stress).
You need Python 3.10–3.12 and Git. Clone this repository and install the required
packages in a virtual environment:

```bash
git clone https://github.com/gianmarcogenalti/combinatorial-bandits-relative-feedback.git
cd combinatorial-bandits-relative-feedback
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all the paper experiments with:

```bash
ews run experiment.yml --output outputs/paper --workers 2
```

EWS runs the simulations, computes regret, and exports the figures. Change
`--workers` to choose the number of parallel workers. To resume an interrupted
experiment, repeat the same command with the same output folder. To inspect its
saved progress, use:

```bash
ews inspect outputs/paper
```

The results are saved in `outputs/paper`. The three TikZ plots are in
`outputs/paper/analysis/figures/`:

- `cumulative_regret.tikz`
- `final_regret_vs_m.tikz`
- `final_regret_vs_theta0.tikz`

The plots inherit the fonts of the LaTeX document in which they are included.
Load `pgfplots`, its `fillbetween` library, and `fix-cm`, and set
`\pgfplotsset{compat=1.18}` before using `\input{PATH/NAME.tikz}`.
To regenerate the plots from compatible saved analysis, use:

```bash
ews plot experiment.yml --output outputs/paper
```

## Experiments

[`experiment.yml`](experiment.yml) contains the instances, algorithm parameters,
and random seed. Each setting uses 20 independent trials of 1,000,000 rounds,
for a total of 360 runs.

| Experiment | Setting | Algorithms |
| --- | --- | --- |
| Ranking feedback | 400 arms, k=m=2; two weights equal to 1 and the others equal to 0.05 | Top-k-UIF and RecMaxMinUCB |
| Feedback level | 16 arms, k=6, m in {1,2,3,6} | Top-k-UIF |
| MNL feedback | 30 arms, k=5, theta0 in {0.001,0.003,0.01,0.03} | MNL-Top-k-UIF, AOA-RB-PL, and AOA-RB-PL-Adaptive |

Figures show mean regret with pointwise 95% Student-t confidence intervals over
the independent trials. The saved aggregate curves, final confidence limits,
and paper figures are also available in [`reference_results/`](reference_results/).
Package versions are pinned in `requirements.txt`; source revisions and result
checksums are recorded in `reference_results/provenance.json`.

## Cite Top-k-UIF

If you use this code in a scientific publication, please cite:

```bibtex
@misc{migali2026ranking,
  title={Worst-Case Regret Bounds for Combinatorial Bandits with Ranking Feedback},
  author={Migali, Cristiano and Genalti, Gianmarco and Metelli, Alberto Maria and Mussi, Marco},
  year={2026}
}
```

## How to Contact Us

You can write an e-mail to cristiano.migali@polimi.it.
