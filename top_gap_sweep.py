# top_gap_sweep.py
#
# Sweep the gap between the best arm (rank 1) and the rest of the top-k
# (ranks 2..k):
#     top_gap = theta_(1) - theta_(2)
# while holding the k-th/(k+1)-th boundary gap fixed, and plot final
# cumulative regret at the horizon T against top_gap for each algorithm.
#
# This is the natural control experiment to run alongside hardness_sweep.py:
# top-k identification only requires resolving the k-th/(k+1)-th boundary
# (set membership), not the internal ranking within the top-k. So, unlike
# hardness_sweep.py's boundary-gap sweep, final regret here is expected to
# stay roughly flat as top_gap shrinks to zero, rather than blowing up.
#
# Uses the "top_gap" instance family (make_top_gap_theta): arm 1 is at
# top_value, arms 2..k are tied at top_value - top_gap, the remaining n-k
# arms are tied at top_value - top_gap - boundary_gap.
#
# Usage:
#     python top_gap_sweep.py --name hardness_gap_sweep \
#         --top-gap-values 0.3 0.2 0.1 0.05 0.02 0.01 0.005
#
# The base experiment's n, k, m, T, delta, num_seeds, algorithms are reused;
# only the instance is replaced per top_gap value.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments import (
    ExperimentConfig,
    config_from_file,
    validate_config,
    run_single_seed,
    make_top_gap_theta,
    theta_diagnostics,
    make_pbar,
    _truncate,
)
from feedback_sensitivity import add_completion_and_tail_metrics


# ============================================================
# Aggregation helpers
# ============================================================

def summarize_by_top_gap(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df
        .groupby(["algorithm", "top_gap"])
        .agg(
            mean_final_regret=("final_cumulative_regret", "mean"),
            std_final_regret=("final_cumulative_regret", "std"),
            median_final_regret=("final_cumulative_regret", "median"),
            correctness_rate=("correct", "mean"),
            completion_rate=("completed", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_completion_round=("completion_round", "mean"),
        )
        .reset_index()
    )


# ============================================================
# Plotting
# ============================================================

# Algorithms hidden from the rebuttal-facing plots/tables by default (still
# recorded in history.csv/summary.csv). Mirrors hardness_sweep.py's default.
DEFAULT_EXCLUDE_ALGORITHMS: tuple[str, ...] = ("uniform",)


def plot_final_regret_vs_top_gap(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    log_x: bool = True,
    exclude_algorithms: Sequence[str] = DEFAULT_EXCLUDE_ALGORITHMS,
    error_mode: str = "sem",
) -> None:
    """
    error_mode: "sem" for +/- 1 standard error of the mean (default), or
    "ci95" for the wider +/- 1.96 * SEM 95% confidence interval.
    """
    if error_mode not in {"sem", "ci95"}:
        raise ValueError("error_mode must be 'sem' or 'ci95'.")
    error_multiplier = 1.96 if error_mode == "ci95" else 1.0

    df = summary_df[~summary_df["algorithm"].isin(exclude_algorithms)]
    grouped = summarize_by_top_gap(df)
    counts = df.groupby(["algorithm", "top_gap"]).size()

    plt.figure(figsize=(7, 5))

    for algorithm, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("top_gap")
        x = sub["top_gap"].to_numpy()
        y = sub["mean_final_regret"].to_numpy()

        n_seeds = np.array([counts[(algorithm, g)] for g in sub["top_gap"]])
        err = error_multiplier * sub["std_final_regret"].fillna(0.0).to_numpy() / np.sqrt(n_seeds)

        plt.errorbar(x, y, yerr=err, marker="o", capsize=4, label=algorithm)

    plt.xlabel(r"Top gap $\theta_{(1)} - \theta_{(2)}$ (boundary gap held fixed)")
    plt.ylabel("Final cumulative top-k regret (at T)")
    plt.title("Final regret vs top-1/top-2 gap (control)")

    if log_x:
        plt.xscale("log")

    # Decreasing top_gap left to right, consistent with hardness_sweep.py.
    plt.gca().invert_xaxis()

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_final_regret_vs_top_gap_boxplot(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    exclude_algorithms: Sequence[str] = DEFAULT_EXCLUDE_ALGORITHMS,
) -> None:
    """
    One panel per algorithm, boxplot of final regret across seeds for each
    top_gap, ordered decreasing left to right.
    """
    df = summary_df[~summary_df["algorithm"].isin(exclude_algorithms)]
    algorithms = sorted(df["algorithm"].unique())
    gaps = sorted(df["top_gap"].unique(), reverse=True)

    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5), squeeze=False)

    for ax, algorithm in zip(axes[0], algorithms):
        data = [
            df.loc[
                (df["algorithm"] == algorithm) & (df["top_gap"] == gap),
                "final_cumulative_regret",
            ].to_numpy()
            for gap in gaps
        ]
        ax.boxplot(data, labels=[f"{g:g}" for g in gaps])
        ax.set_xlabel(r"$\theta_{(1)} - \theta_{(2)}$")
        ax.set_ylabel("Final cumulative top-k regret")
        ax.set_title(algorithm)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# Core sweep
# ============================================================

def run_top_gap_sweep(
    *,
    base_config: ExperimentConfig,
    top_gap_values: Iterable[float],
    output_dir: Path,
    top_value: float = 1.0,
    boundary_gap: float = 0.2,
    tail_value: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: list[pd.DataFrame] = []
    summaries: list[dict] = []

    top_gap_values = sorted(set(float(g) for g in top_gap_values), reverse=True)

    print("Base experiment:", base_config.name)
    print("Testing top gaps (theta_1 - theta_2):", top_gap_values)
    print("Fixed boundary gap:", boundary_gap)
    print()

    total_runs = len(top_gap_values) * len(base_config.algorithms) * base_config.num_seeds
    outer = make_pbar(
        total=total_runs,
        desc=_truncate(f"{base_config.name} (by top_gap)", 28),
        unit="run",
    )

    try:
        for top_gap in top_gap_values:
            theta = make_top_gap_theta(
                n=base_config.n,
                k=base_config.k,
                top_value=top_value,
                top_gap=top_gap,
                boundary_gap=boundary_gap,
                tail_value=tail_value,
                shuffle=False,
            )

            tqdm.write(
                f"top_gap={top_gap:.4f}  theta_diagnostics={theta_diagnostics(theta, k=base_config.k)}"
            )

            sweep_config = replace(
                base_config,
                name=f"{base_config.name}_topgap{top_gap:g}",
                output_dir=str(output_dir),
            )

            for algorithm_name in base_config.algorithms:
                for seed in range(base_config.num_seeds):
                    history_df, summary = run_single_seed(
                        algorithm_name=algorithm_name,
                        theta=theta,
                        config=sweep_config,
                        seed=seed,
                    )

                    history_df = history_df.copy()
                    history_df["top_gap"] = top_gap

                    summary = add_completion_and_tail_metrics(
                        history_df=history_df,
                        summary=summary,
                        T=sweep_config.T,
                    )
                    summary["top_gap"] = top_gap

                    histories.append(history_df)
                    summaries.append(summary)

                    tqdm.write(
                        f"top_gap={top_gap:.4f} {algorithm_name:16s} seed={seed:03d} "
                        f"final_regret={summary['final_cumulative_regret']:.4f} "
                        f"correct={summary['correct']} completed={summary['completed']}"
                    )

                    outer.update(1)
    finally:
        outer.close()

    history_all = pd.concat(histories, ignore_index=True)
    summary_all = pd.DataFrame(summaries)

    return history_all, summary_all


# ============================================================
# Save outputs
# ============================================================

def save_outputs(
    *,
    base_config: ExperimentConfig,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_gap_values: list[float],
    boundary_gap: float,
    output_dir: Path,
    exclude_algorithms: Sequence[str] = DEFAULT_EXCLUDE_ALGORITHMS,
    error_mode: str = "sem",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_dir / "history.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    aggregate_summary = summarize_by_top_gap(summary_df)
    aggregate_summary.to_csv(output_dir / "aggregate_summary_by_top_gap.csv", index=False)

    metadata = {
        "base_config": asdict(base_config),
        "top_gap_values": sorted(top_gap_values, reverse=True),
        "boundary_gap": boundary_gap,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plot_final_regret_vs_top_gap(
        summary_df,
        output_path=output_dir / "final_regret_vs_top_gap.png",
        exclude_algorithms=exclude_algorithms,
        error_mode=error_mode,
    )

    plot_final_regret_vs_top_gap_boxplot(
        summary_df,
        output_path=output_dir / "final_regret_vs_top_gap_boxplot.png",
        exclude_algorithms=exclude_algorithms,
    )

    print()
    print("Aggregate summary by top_gap:")
    print(aggregate_summary.to_string(index=False))

    print()
    print("Saved outputs to:", output_dir)
    print("  history.csv")
    print("  summary.csv")
    print("  aggregate_summary_by_top_gap.csv")
    print("  metadata.json")
    print("  final_regret_vs_top_gap.png")
    print("  final_regret_vs_top_gap_boxplot.png")
    print()

    exclude_args = " ".join(exclude_algorithms)
    print("To build a rebuttal table (rows=top_gap, cols=algorithm):")
    print(
        f"  python make_markdown_tables.py --results-dir {output_dir} "
        "--pivot-metric final_cumulative_regret --rows top_gap --cols algorithm "
        f"--error {error_mode}" + (f" --exclude {exclude_args}" if exclude_args else "")
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="experiment_configs.yaml",
        help="Path to YAML/JSON experiment config file.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the base experiment config (its n, k, m, T, delta, "
        "num_seeds, algorithms are reused; its instance is replaced per top_gap).",
    )

    parser.add_argument(
        "--top-gap-values",
        type=float,
        nargs="+",
        default=None,
        help="theta_(1) - theta_(2) values to test. "
        "Default: 0.4 0.3 0.2 0.1 0.05 0.02 0.01 0.005",
    )

    parser.add_argument(
        "--top-value",
        type=float,
        default=1.0,
        help="theta value of the single best arm (rank 1).",
    )

    parser.add_argument(
        "--boundary-gap",
        type=float,
        default=0.2,
        help="Fixed k-th/(k+1)-th boundary gap, held constant across the sweep. "
        "Should be large enough that top-k identification stays easy on its own "
        "(0.2 completes quickly per hardness_sweep.py's calibration at n=10,k=3).",
    )

    parser.add_argument(
        "--tail-value",
        type=float,
        default=None,
        help="Common theta value for the non-top-k arms. "
        "Default: top_value - top_gap - boundary_gap for each top_gap.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save results. Default: results/<name>_top_gap_sweep.",
    )

    parser.add_argument(
        "--exclude-algorithms",
        type=str,
        nargs="*",
        default=list(DEFAULT_EXCLUDE_ALGORITHMS),
        help="Algorithms to hide from the plots (still recorded in history.csv/"
        f"summary.csv). Default: {list(DEFAULT_EXCLUDE_ALGORITHMS)}. Pass "
        "--exclude-algorithms with no values to include everything.",
    )

    parser.add_argument(
        "--error-mode",
        type=str,
        choices=["sem", "ci95"],
        default="sem",
        help="Error bars: 'sem' for +/- 1 standard error of the mean (default), "
        "or 'ci95' for the wider +/- 1.96*SEM 95%% confidence interval.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_config = config_from_file(Path(args.config), args.name)
    validate_config(base_config)

    if base_config.k < 2:
        raise ValueError(
            "top_gap_sweep.py requires k >= 2 (need at least 2 top arms to "
            "have a distinct rank-1/rank-2 gap)."
        )

    top_gap_values = args.top_gap_values or [0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    for gap in top_gap_values:
        if not (0.0 < gap < args.top_value):
            raise ValueError(f"Invalid top_gap value {gap}: must be in (0, top_value={args.top_value}).")

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f"results/{base_config.name}_top_gap_sweep"
    )

    history_df, summary_df = run_top_gap_sweep(
        base_config=base_config,
        top_gap_values=top_gap_values,
        output_dir=output_dir,
        top_value=args.top_value,
        boundary_gap=args.boundary_gap,
        tail_value=args.tail_value,
    )

    save_outputs(
        base_config=base_config,
        history_df=history_df,
        summary_df=summary_df,
        top_gap_values=top_gap_values,
        boundary_gap=args.boundary_gap,
        output_dir=output_dir,
        exclude_algorithms=tuple(args.exclude_algorithms),
        error_mode=args.error_mode,
    )


if __name__ == "__main__":
    main()
