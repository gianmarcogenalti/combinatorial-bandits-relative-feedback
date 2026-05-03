from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import (
    ExperimentConfig,
    config_from_file,
    validate_config,
    run_single_seed,
    build_theta,
    theta_diagnostics,
)


# ============================================================
# Aggregation helpers
# ============================================================

def aggregate_curve(
    history_df: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    grouped = (
        history_df
        .groupby(["feedback_m", "round"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    return grouped


def add_moving_average(
    history_df: pd.DataFrame,
    *,
    value_col: str,
    output_col: str,
    window: int,
) -> pd.DataFrame:
    df = history_df.sort_values(["feedback_m", "seed", "round"]).copy()

    df[output_col] = (
        df
        .groupby(["feedback_m", "seed"])[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def summarize_by_m(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df
        .groupby("feedback_m")
        .agg(
            mean_final_regret=("final_cumulative_regret", "mean"),
            std_final_regret=("final_cumulative_regret", "std"),
            median_final_regret=("final_cumulative_regret", "median"),
            correctness_rate=("correct", "mean"),
            completion_rate=("completed", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_completion_round=("completion_round", "mean"),
            mean_tail_instant_regret=("tail_mean_instantaneous_regret", "mean"),
        )
        .reset_index()
    )


# ============================================================
# Extra summaries
# ============================================================

def add_completion_and_tail_metrics(
    *,
    history_df: pd.DataFrame,
    summary: dict,
    T: int,
) -> dict:
    """
    Add completion round and tail instantaneous regret to one seed summary.
    """

    out = dict(summary)

    completed_rows = history_df[history_df["completed"] == True]  # noqa: E712

    if len(completed_rows) > 0:
        out["completion_round"] = int(completed_rows["round"].iloc[0])
    else:
        out["completion_round"] = np.nan

    tail_start = int(0.9 * T)
    tail = history_df[history_df["round"] >= tail_start]

    if len(tail) > 0:
        out["tail_mean_instantaneous_regret"] = float(
            tail["instantaneous_regret"].mean()
        )
        out["tail_mean_cumulative_regret"] = float(
            tail["cumulative_regret"].mean()
        )
    else:
        out["tail_mean_instantaneous_regret"] = np.nan
        out["tail_mean_cumulative_regret"] = np.nan

    return out


# ============================================================
# Plotting
# ============================================================

def plot_cumulative_regret_by_m(
    history_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    agg = aggregate_curve(history_df, value_col="cumulative_regret")

    plt.figure(figsize=(8, 5))

    for m, sub in agg.groupby("feedback_m"):
        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        plt.plot(x, y, label=f"m={m}")
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel("Cumulative top-k regret")
    plt.title("TOP-k-UIF sensitivity to feedback amount")
    plt.legend(title="Feedback")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_instantaneous_regret_by_m(
    history_df: pd.DataFrame,
    *,
    output_path: Path,
    window: int,
) -> None:
    df = add_moving_average(
        history_df,
        value_col="instantaneous_regret",
        output_col="instantaneous_regret_ma",
        window=window,
    )

    agg = aggregate_curve(df, value_col="instantaneous_regret_ma")

    plt.figure(figsize=(8, 5))

    for m, sub in agg.groupby("feedback_m"):
        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        plt.plot(x, y, label=f"m={m}")
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel(f"Instantaneous top-k regret, moving average window={window}")
    plt.title("Instantaneous regret vs feedback amount")
    plt.legend(title="Feedback")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_final_regret_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    grouped = summarize_by_m(summary_df)

    x = grouped["feedback_m"].to_numpy()
    y = grouped["mean_final_regret"].to_numpy()
    err = 1.96 * grouped["std_final_regret"].fillna(0.0).to_numpy() / np.sqrt(
        summary_df.groupby("feedback_m").size().to_numpy()
    )

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=err, marker="o", capsize=4)
    plt.xlabel("Feedback level m")
    plt.ylabel("Final cumulative top-k regret")
    plt.title("Final regret vs feedback amount")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_success_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    grouped = summarize_by_m(summary_df)

    x = grouped["feedback_m"].to_numpy()
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(
        x - width / 2,
        grouped["correctness_rate"],
        width,
        label="correct",
    )
    plt.bar(
        x + width / 2,
        grouped["completion_rate"],
        width,
        label="completed",
    )

    plt.xlabel("Feedback level m")
    plt.ylabel("Fraction of seeds")
    plt.title("Identification success vs feedback amount")
    plt.xticks(x)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_completion_round_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    grouped = summarize_by_m(summary_df)

    plt.figure(figsize=(7, 5))
    plt.plot(
        grouped["feedback_m"],
        grouped["mean_completion_round"],
        marker="o",
    )
    plt.xlabel("Feedback level m")
    plt.ylabel("Mean completion round")
    plt.title("Completion time vs feedback amount")
    plt.xticks(grouped["feedback_m"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Core sweep
# ============================================================

def run_feedback_sensitivity(
    *,
    base_config: ExperimentConfig,
    m_values: Iterable[int],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    theta = build_theta(base_config)

    histories: list[pd.DataFrame] = []
    summaries: list[dict] = []

    print("Base experiment:", base_config.name)
    print("Testing feedback levels:", list(m_values))
    print("Theta diagnostics:", theta_diagnostics(theta, k=base_config.k))
    print("Theta:", np.round(theta, 4))
    print()

    for m in m_values:
        sweep_config = replace(
            base_config,
            name=f"{base_config.name}_m{m}",
            m=int(m),
            algorithms=("topk_uif",),
            output_dir=str(output_dir),
        )

        validate_config(sweep_config)

        for seed in range(sweep_config.num_seeds):
            history_df, summary = run_single_seed(
                algorithm_name="topk_uif",
                theta=theta,
                config=sweep_config,
                seed=seed,
            )

            history_df = history_df.copy()
            history_df["base_experiment"] = base_config.name
            history_df["feedback_m"] = int(m)
            history_df["feedback_fraction"] = float(m / base_config.k)

            summary = add_completion_and_tail_metrics(
                history_df=history_df,
                summary=summary,
                T=sweep_config.T,
            )
            summary["base_experiment"] = base_config.name
            summary["feedback_m"] = int(m)
            summary["feedback_fraction"] = float(m / base_config.k)

            histories.append(history_df)
            summaries.append(summary)

            print(
                f"m={m:2d} seed={seed:03d} "
                f"final_regret={summary['final_cumulative_regret']:.4f} "
                f"correct={summary['correct']} "
                f"completed={summary['completed']} "
                f"completion_round={summary['completion_round']}"
            )

    history_all = pd.concat(histories, ignore_index=True)
    summary_all = pd.DataFrame(summaries)

    return history_all, summary_all, theta


# ============================================================
# Save outputs
# ============================================================

def save_outputs(
    *,
    base_config: ExperimentConfig,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    theta: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_dir / "history.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    aggregate_summary = summarize_by_m(summary_df)
    aggregate_summary.to_csv(output_dir / "aggregate_summary_by_m.csv", index=False)

    np.savetxt(output_dir / "theta.csv", theta, delimiter=",")

    metadata = {
        "base_config": asdict(base_config),
        "theta_diagnostics": theta_diagnostics(theta, k=base_config.k),
        "theta": theta.tolist(),
        "m_values": sorted(summary_df["feedback_m"].unique().tolist()),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    num_saved_points = history_df["round"].nunique()
    instant_window = max(1, min(500, num_saved_points // 30))

    plot_cumulative_regret_by_m(
        history_df,
        output_path=output_dir / "cumulative_regret_by_m.png",
    )

    plot_instantaneous_regret_by_m(
        history_df,
        output_path=output_dir / "instantaneous_regret_by_m.png",
        window=instant_window,
    )

    plot_final_regret_vs_m(
        summary_df,
        output_path=output_dir / "final_regret_vs_m.png",
    )

    plot_success_vs_m(
        summary_df,
        output_path=output_dir / "success_vs_m.png",
    )

    plot_completion_round_vs_m(
        summary_df,
        output_path=output_dir / "completion_round_vs_m.png",
    )

    print()
    print("Aggregate summary by m:")
    print(aggregate_summary.to_string(index=False))

    print()
    print("Saved outputs to:", output_dir)
    print("  history.csv")
    print("  summary.csv")
    print("  aggregate_summary_by_m.csv")
    print("  theta.csv")
    print("  metadata.json")
    print("  cumulative_regret_by_m.png")
    print("  instantaneous_regret_by_m.png")
    print("  final_regret_vs_m.png")
    print("  success_vs_m.png")
    print("  completion_round_vs_m.png")


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
        help="Name of the base experiment config.",
    )

    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=None,
        help="Feedback levels to test. Default: all m=1,...,k.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save results. Default: results/<name>_feedback_sensitivity.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_config = config_from_file(Path(args.config), args.name)

    if args.m_values is None:
        m_values = list(range(1, base_config.k + 1))
    else:
        m_values = sorted(set(args.m_values))

    for m in m_values:
        if not (1 <= m <= base_config.k):
            raise ValueError(f"Invalid m={m}. Require 1 <= m <= k={base_config.k}.")

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f"results/{base_config.name}_feedback_sensitivity"
    )

    history_df, summary_df, theta = run_feedback_sensitivity(
        base_config=base_config,
        m_values=m_values,
        output_dir=output_dir,
    )

    save_outputs(
        base_config=base_config,
        history_df=history_df,
        summary_df=summary_df,
        theta=theta,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()