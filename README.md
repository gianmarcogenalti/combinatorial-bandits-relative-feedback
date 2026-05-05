# Experiments

This repository runs experiments from a YAML configuration file. The default config file is:

```bash
experiment_configs.yaml
```

Each YAML file contains one or more named experiments. To launch an experiment, use:

```bash
python experiments.py --name <experiment_name>
```

For example:

```bash
python experiments.py --name weak_kth_pathological
```

To list all available experiments:

```bash
python experiments.py --list
```

To use a different YAML file:

```bash
python experiments.py --config my_experiments.yaml --name weak_kth_pathological
```

## Example YAML config

```yaml
experiments:
  weak_kth_pathological:
    n: 10
    k: 5
    m: 5
    T: 100000
    num_seeds: 10
    delta: 0.05

    algorithms:
      - topk_uif
      - rec_maxmin_ucb

    instance:
      kind: weak_kth
      strong_value: 1.0
      weak_optimal_value: 0.52
      decoy_value: 0.50
      tail_value: 0.15
      num_decoys: 5
      shuffle: true
      seed: 123

    output_dir: results/weak_kth_pathological
    save_every: 10

    recmaxmin_alpha: 0.51
    allow_recmaxmin_topm_variant: false
```

## Main parameters

`n` is the total number of arms.

`k` is the number of arms pulled at each round. The goal is to learn or play the best `k` arms.

`m` is the amount of ranking feedback observed after pulling `k` arms. The parameters must satisfy:

```text
1 <= m <= k <= n
```

If `m = 1`, the learner observes only the winner among the pulled arms. If `m = k`, the learner observes the full ranking of the pulled arms.

Faithful `rec_maxmin_ucb` requires full-ranking feedback, so use:

```yaml
m: k
```

or, concretely, if `k: 5`:

```yaml
m: 5
```

`T` is the number of rounds.

`num_seeds` is the number of independent repetitions. Use multiple seeds to obtain meaningful confidence intervals.

`delta` is the confidence parameter used by `topk_uif`. Smaller values make the algorithm more conservative.

`algorithms` is the list of algorithms to run. Available options are:

```text
topk_uif
rec_maxmin_ucb
uniform
```

`output_dir` is the folder where results are saved.

`save_every` controls how often one row is saved in `history.csv`. For example, `save_every: 10` stores one row every 10 rounds. For large horizons, use a larger value such as `save_every: 1000`.

## Instance types

The bandit instance is specified under the `instance` field.

### Explicit instance

Use `kind: explicit` to manually specify all Plackett-Luce parameters.

```yaml
instance:
  kind: explicit
  theta:
    - 1.00
    - 0.90
    - 0.80
    - 0.50
    - 0.40
```

The length of `theta` must equal `n`. Larger values correspond to better arms.

### Weak kth-arm instance

Use `kind: weak_kth` for an instance with `k-1` strong optimal arms, one weak optimal arm, and several near-optimal decoys.

```yaml
instance:
  kind: weak_kth
  strong_value: 1.0
  weak_optimal_value: 0.52
  decoy_value: 0.50
  tail_value: 0.15
  num_decoys: 5
  shuffle: true
  seed: 123
```

The fields mean:

```text
strong_value         value of the first k-1 optimal arms
weak_optimal_value   value of the kth optimal arm
decoy_value          value of near-optimal suboptimal arms
tail_value           value of remaining bad arms
num_decoys           number of near-optimal decoys
shuffle              whether to randomly permute arm identities
seed                 seed used for the shuffle
```

`num_decoys` must satisfy:

```text
num_decoys <= n - k
```

### Boundary-cloud instance

Use `kind: boundary_cloud` to create many arms near the top-k boundary.

```yaml
instance:
  kind: boundary_cloud
  top_start: 1.0
  boundary_value: 0.50
  gap: 0.05
  cloud_size: 10
  tail_value: 0.20
  shuffle: true
  seed: 123
```

### Two-level decoy instance

Use `kind: two_level_decoy` to create both close decoys and clearly bad decoys.

```yaml
instance:
  kind: two_level_decoy
  strong_value: 1.0
  weak_optimal_value: 0.60
  close_decoy_value: 0.55
  bad_decoy_value: 0.20
  close_decoy_fraction: 0.5
  shuffle: true
  seed: 123
```

## Rec-MaxMin-UCB parameters

`recmaxmin_alpha` is the exploration parameter for `rec_maxmin_ucb`.

```yaml
recmaxmin_alpha: 0.51
```

`allow_recmaxmin_topm_variant` controls whether Rec-MaxMin-UCB can be run with partial top-m feedback.

```yaml
allow_recmaxmin_topm_variant: false
```

If this is `false`, `rec_maxmin_ucb` is allowed only when `m = k`.

If this is `true`, the code allows a partial-feedback variant, but this is not the faithful baseline.

## Feedback sensitivity experiments

To test only `topk_uif` with different feedback levels `m = 1, ..., k`, run:

```bash
python feedback_sensitivity.py --name <experiment_name>
```

For example:

```bash
python feedback_sensitivity.py --name weak_kth_pathological
```

To test only selected feedback levels:

```bash
python feedback_sensitivity.py --name weak_kth_pathological --m-values 1 2 5
```

The output is saved by default to:

```text
results/<experiment_name>_feedback_sensitivity/
```

## Generating PDF plots

After an experiment has finished, generate PDF plots without rerunning the experiment:

```bash
python make_pdf_plots.py --results-dir results/weak_kth_pathological
```

To exclude an algorithm from the plots:

```bash
python make_pdf_plots.py --results-dir results/weak_kth_pathological --exclude uniform
```

To downsample large plots, for example keeping one plotted point every 1000 rounds:

```bash
python make_pdf_plots.py --results-dir results/weak_kth_pathological --stride 1000
```

PDF plots are saved in:

```text
results/<experiment_name>/pdf_plots/

```

## MNL top-k experiments

The repository also supports MNL bandit experiments with a no-choice item. These experiments compare:

```text
mnl_topk_uif    MNL version of TOP-k-UIF
sg_adaptive     Saha-Gaillard adaptive-pivot baseline
sg_no_choice    Saha-Gaillard no-choice-pivot baseline
```

A typical YAML config is:

```yaml
experiments:
  mnl_topk_theta0_sweep:
    task: mnl_topk
    n: 30
    k: 5
    T: 100000
    num_seeds: 20
    delta: 0.05

    algorithms:
      - mnl_topk_uif
      - sg_adaptive
      - sg_no_choice

    theta0_values:
      - 1.0
      - 0.1
      - 0.01
      - 0.001

    topk_uif_beta_scale: 0.1
    sg_x_scale: 2.0

    instance:
      kind: explicit
      theta:
        - 1.00
        - 0.95
        - 0.90
        - 0.85
        - 0.80
        - 0.65
        - 0.62
        - 0.59
        - 0.56
        - 0.53
        - 0.50
        - 0.47
        - 0.44
        - 0.41
        - 0.38
        - 0.35
        - 0.32
        - 0.29
        - 0.26
        - 0.23
        - 0.20
        - 0.18
        - 0.16
        - 0.14
        - 0.12
        - 0.10
        - 0.08
        - 0.06
        - 0.04
        - 0.02

    output_dir: results/mnl_topk_theta0_sweep
    save_every: 100
```

### Meaning of the MNL fields

`task: mnl_topk` tells the code to run the MNL top-k experiment.

`theta0_values` is the list of no-choice parameters to test. Smaller values mean the no-choice option is selected less often.

```yaml
theta0_values:
  - 1.0
  - 0.1
  - 0.01
  - 0.001
```

`topk_uif_beta_scale` rescales the confidence radius of `mnl_topk_uif`. Smaller values make the algorithm less conservative.

```yaml
topk_uif_beta_scale: 0.1
```

`sg_x_scale` controls the exploration parameter used by the Saha-Gaillard baselines. The code uses approximately:

```text
x = sg_x_scale * log(T)
```

Typical value:

```yaml
sg_x_scale: 2.0
```

### Run the MNL experiment

```bash
python mnl_experiments.py --name mnl_topk_theta0_sweep
```

Use a different config file:

```bash
python mnl_experiments.py --config my_experiments.yaml --name mnl_topk_theta0_sweep
```

For large horizons, downsample the plotted curves:

```bash
python mnl_experiments.py --name mnl_topk_theta0_sweep --stride 1000
```

The results are saved in:

```text
results/mnl_topk_theta0_sweep/
```

The PDF plots are saved in:

```text
results/mnl_topk_theta0_sweep/pdf_plots/
```

Typical generated plots are:

```text
cumulative_regret_theta0_1.pdf
cumulative_regret_theta0_0.1.pdf
cumulative_regret_theta0_0.01.pdf
cumulative_regret_theta0_0.001.pdf
instantaneous_regret_theta0_1.pdf
instantaneous_regret_theta0_0.1.pdf
instantaneous_regret_theta0_0.01.pdf
instantaneous_regret_theta0_0.001.pdf
final_regret_vs_theta0.pdf
tail_instantaneous_regret_vs_theta0.pdf
last_action_success_vs_theta0.pdf
```

### Recommended commands

Run the full theta0 sweep:

```bash
python mnl_experiments.py --name mnl_topk_theta0_sweep --stride 1000
```

Run only one no-choice value by editing the YAML:

```yaml
theta0_values:
  - 0.01
```

Then run:

```bash
python mnl_experiments.py --name mnl_topk_theta0_sweep --stride 1000
```

Compare only the two Saha-Gaillard baselines by editing the YAML:

```yaml
algorithms:
  - sg_adaptive
  - sg_no_choice
```

Compare only MNL-TOP-k-UIF with the adaptive baseline:

```yaml
algorithms:
  - mnl_topk_uif
  - sg_adaptive
```

### Interpreting the MNL experiment

When `theta0 = 1`, the no-choice item is not rare, so `sg_adaptive` and `sg_no_choice` may behave similarly.

When `theta0` is small, no-choice is rarely selected. In this regime, `sg_no_choice` can perform poorly because it estimates every arm through direct comparisons with no-choice. The adaptive baseline should improve by using intermediate pivots, while `mnl_topk_uif` explicitly changes the played set to move away from the no-choice pivot.

The most important plots are:

```text
final_regret_vs_theta0.pdf
tail_instantaneous_regret_vs_theta0.pdf
```

## Output files

Each experiment produces files such as:

```text
history.csv
summary.csv
aggregate_summary.csv
theta.csv
metadata.json
cumulative_regret.png
instantaneous_regret.png
```

`history.csv` contains per-round trajectories such as regret, accepted set size, unknown set size, and completion status.

`summary.csv` contains one row per algorithm and seed.

`aggregate_summary.csv` contains metrics aggregated across seeds.

`theta.csv` contains the Plackett-Luce parameter vector used in the experiment.

`metadata.json` stores the experiment config and diagnostics.

## Notes

Use more than one seed if you want meaningful confidence intervals. The confidence bands in the plots are computed across seeds as:

```text
mean ± std / sqrt(num_seeds)
```

For large experiments, increase `save_every` to reduce file size and plotting time.