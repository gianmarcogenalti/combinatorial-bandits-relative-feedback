# hardness_sweep.py
#
# Sweep the instance hardness by shrinking the boundary gap
#     delta_min = theta_(k) - theta_(k+1)
# between the k-th and (k+1)-th arm, and plot final cumulative regret at the
# horizon T against delta_min for each algorithm.
#
# Uses the "gap" instance family (make_gap_theta): the top-k arms are tied at
# top_value, the remaining n-k arms are tied at top_value - delta_min. This
# isolates the boundary gap as the only lever, so it is a clean hardness knob.
#
# Usage:
#     python hardness_sweep.py --name weak_kth_easier \
#         --gap-values 0.4 0.3 0.2 0.15 0.1 0.07 0.05 0.03 0.02 0.01
#
# The base experiment's n, k, m, T, delta, num_seeds, algorithms are reused;
# only the instance is replaced per gap value.

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments import (
    ExperimentConfig,
    config_from_file,
    validate_config,
    run_single_seed,
    make_gap_theta,
    theta_diagnostics,
    make_pbar,
    _truncate,
)
from feedback_sensitivity import add_completion_and_tail_metrics


# ============================================================
# Aggregation helpers
# ============================================================

def summarize_by_gap(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df
        .groupby(["algorithm", "delta_min"])
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
# recorded in history.csv/summary.csv). uniform's regret scales in the
# opposite direction (it grows with delta_min, since it never adapts), which
# swamps the topk_uif vs rec_maxmin_ucb comparison on a shared axis.
DEFAULT_EXCLUDE_ALGORITHMS: tuple[str, ...] = ("uniform",)


def plot_final_regret_vs_delta_min(
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
    grouped = summarize_by_gap(df)
    counts = df.groupby(["algorithm", "delta_min"]).size()

    plt.figure(figsize=(7, 5))

    for algorithm, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("delta_min")
        x = sub["delta_min"].to_numpy()
        y = sub["mean_final_regret"].to_numpy()

        n_seeds = np.array([counts[(algorithm, g)] for g in sub["delta_min"]])
        err = error_multiplier * sub["std_final_regret"].fillna(0.0).to_numpy() / np.sqrt(n_seeds)

        plt.errorbar(x, y, yerr=err, marker="o", capsize=4, label=algorithm)

    plt.xlabel(r"Boundary gap $\Delta_{\min} = \theta_{(k)} - \theta_{(k+1)}$")
    plt.ylabel("Final cumulative top-k regret (at T)")
    plt.title("Final regret vs instance hardness")

    if log_x:
        plt.xscale("log")

    # Decreasing delta_min left to right: increasing hardness reads left to right.
    plt.gca().invert_xaxis()

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_final_regret_vs_delta_min_boxplot(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    exclude_algorithms: Sequence[str] = DEFAULT_EXCLUDE_ALGORITHMS,
) -> None:
    """
    One panel per algorithm, boxplot of final regret across seeds for each
    gap, with gaps ordered decreasing left to right (increasing hardness).
    """
    df = summary_df[~summary_df["algorithm"].isin(exclude_algorithms)]
    algorithms = sorted(df["algorithm"].unique())
    gaps = sorted(df["delta_min"].unique(), reverse=True)

    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5), squeeze=False)

    for ax, algorithm in zip(axes[0], algorithms):
        data = [
            df.loc[
                (df["algorithm"] == algorithm) & (df["delta_min"] == gap),
                "final_cumulative_regret",
            ].to_numpy()
            for gap in gaps
        ]
        ax.boxplot(data, labels=[f"{g:g}" for g in gaps])
        ax.set_xlabel(r"$\Delta_{\min}$")
        ax.set_ylabel("Final cumulative top-k regret")
        ax.set_title(algorithm)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# Core sweep
# ============================================================

def _run_one_job(job: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Runs a single (gap, algorithm, seed) job. Module-level (not a closure) so
    it can be pickled and sent to worker processes by ProcessPoolExecutor.
    Progress is reported at the job level by the caller, not per-round, since
    per-round tqdm bars from multiple processes would interleave on screen.
    """
    history_df, summary = run_single_seed(
        algorithm_name=job["algorithm_name"],
        theta=job["theta"],
        config=job["sweep_config"],
        seed=job["seed"],
        show_progress=False,
    )

    history_df = history_df.copy()
    history_df["delta_min"] = job["gap"]

    summary = add_completion_and_tail_metrics(
        history_df=history_df,
        summary=summary,
        T=job["sweep_config"].T,
    )
    summary["delta_min"] = job["gap"]

    return history_df, summary


def run_hardness_sweep(
    *,
    base_config: ExperimentConfig,
    gap_values: Iterable[float],
    output_dir: Path,
    top_value: float = 1.0,
    tail_value: Optional[float] = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    n_jobs > 1 runs the independent (gap, algorithm, seed) runs in parallel
    worker processes (each run only depends on its own seed, so results are
    identical to the sequential n_jobs=1 case, just faster).
    """
    gap_values = sorted(set(float(g) for g in gap_values), reverse=True)

    print("Base experiment:", base_config.name)
    print("Testing boundary gaps (delta_min):", gap_values)
    print()

    jobs: list[dict[str, Any]] = []
    for gap in gap_values:
        theta = make_gap_theta(
            n=base_config.n,
            k=base_config.k,
            top_value=top_value,
            boundary_gap=gap,
            tail_value=tail_value,
            shuffle=False,
        )

        print(f"delta_min={gap:.4f}  theta_diagnostics={theta_diagnostics(theta, k=base_config.k)}")

        sweep_config = replace(
            base_config,
            name=f"{base_config.name}_gap{gap:g}",
            output_dir=str(output_dir),
        )

        for algorithm_name in base_config.algorithms:
            for seed in range(base_config.num_seeds):
                jobs.append(
                    {
                        "gap": gap,
                        "algorithm_name": algorithm_name,
                        "theta": theta,
                        "sweep_config": sweep_config,
                        "seed": seed,
                    }
                )
    print()

    histories: list[pd.DataFrame] = []
    summaries: list[dict] = []

    outer = make_pbar(
        total=len(jobs),
        desc=_truncate(f"{base_config.name} (by delta_min)", 28),
        unit="run",
    )

    def _report(job: dict[str, Any], summary: dict) -> None:
        tqdm.write(
            f"delta_min={job['gap']:.4f} {job['algorithm_name']:16s} seed={job['seed']:03d} "
            f"final_regret={summary['final_cumulative_regret']:.4f} "
            f"correct={summary['correct']} completed={summary['completed']}"
        )
        outer.update(1)

    try:
        if n_jobs <= 1:
            for job in jobs:
                history_df, summary = _run_one_job(job)
                histories.append(history_df)
                summaries.append(summary)
                _report(job, summary)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_to_job = {executor.submit(_run_one_job, job): job for job in jobs}
                for future in as_completed(future_to_job):
                    history_df, summary = future.result()
                    histories.append(history_df)
                    summaries.append(summary)
                    _report(future_to_job[future], summary)
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
    gap_values: list[float],
    output_dir: Path,
    exclude_algorithms: Sequence[str] = DEFAULT_EXCLUDE_ALGORITHMS,
    error_mode: str = "sem",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_dir / "history.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    aggregate_summary = summarize_by_gap(summary_df)
    aggregate_summary.to_csv(output_dir / "aggregate_summary_by_gap.csv", index=False)

    metadata = {
        "base_config": asdict(base_config),
        "gap_values": sorted(gap_values, reverse=True),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plot_final_regret_vs_delta_min(
        summary_df,
        output_path=output_dir / "final_regret_vs_delta_min.png",
        exclude_algorithms=exclude_algorithms,
        error_mode=error_mode,
    )

    plot_final_regret_vs_delta_min_boxplot(
        summary_df,
        output_path=output_dir / "final_regret_vs_delta_min_boxplot.png",
        exclude_algorithms=exclude_algorithms,
    )

    print()
    print("Aggregate summary by gap:")
    print(aggregate_summary.to_string(index=False))

    print()
    print("Saved outputs to:", output_dir)
    print("  history.csv")
    print("  summary.csv")
    print("  aggregate_summary_by_gap.csv")
    print("  metadata.json")
    print("  final_regret_vs_delta_min.png")
    print("  final_regret_vs_delta_min_boxplot.png")
    print()

    exclude_args = " ".join(exclude_algorithms)
    print("To build a rebuttal table (rows=delta_min, cols=algorithm):")
    print(
        f"  python make_markdown_tables.py --results-dir {output_dir} "
        "--pivot-metric final_cumulative_regret --rows delta_min --cols algorithm "
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
        "num_seeds, algorithms are reused; its instance is replaced per gap).",
    )

    parser.add_argument(
        "--gap-values",
        type=float,
        nargs="+",
        default=None,
        help="Boundary gap (delta_min) values to test. "
        "Default: 0.4 0.3 0.2 0.15 0.1 0.07 0.05 0.03 0.02 0.01",
    )

    parser.add_argument(
        "--top-value",
        type=float,
        default=1.0,
        help="Common theta value for the top-k arms.",
    )

    parser.add_argument(
        "--tail-value",
        type=float,
        default=None,
        help="Common theta value for the non-top-k arms. "
        "Default: top_value - delta_min for each gap.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save results. Default: results/<name>_hardness_sweep.",
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

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes to run (gap, algorithm, seed) runs in "
        "parallel. Each run only depends on its own seed, so results are "
        "identical to --jobs 1, just faster. Default: 1 (sequential).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_config = config_from_file(Path(args.config), args.name)
    validate_config(base_config)

    gap_values = args.gap_values or [0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]

    for gap in gap_values:
        if not (0.0 < gap < args.top_value):
            raise ValueError(f"Invalid gap value {gap}: must be in (0, top_value={args.top_value}).")

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f"results/{base_config.name}_hardness_sweep"
    )

    history_df, summary_df = run_hardness_sweep(
        base_config=base_config,
        gap_values=gap_values,
        output_dir=output_dir,
        top_value=args.top_value,
        tail_value=args.tail_value,
        n_jobs=args.jobs,
    )

    save_outputs(
        base_config=base_config,
        history_df=history_df,
        summary_df=summary_df,
        gap_values=gap_values,
        output_dir=output_dir,
        exclude_algorithms=tuple(args.exclude_algorithms),
        error_mode=args.error_mode,
    )


if __name__ == "__main__":
    main()
