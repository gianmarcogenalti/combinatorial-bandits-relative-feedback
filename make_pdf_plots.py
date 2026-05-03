# make_pdf_plots.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate_curve(df: pd.DataFrame, *, group_col: str, value_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby([group_col, "round"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    return grouped


def add_moving_average(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    output_col: str,
    window: int,
) -> pd.DataFrame:
    df = df.sort_values([group_col, "seed", "round"]).copy()

    df[output_col] = (
        df.groupby([group_col, "seed"])[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def plot_cumulative_regret(
    history_df: pd.DataFrame,
    *,
    group_col: str,
    output_path: Path,
    stride: int
) -> None:
    agg = aggregate_curve(
        history_df,
        group_col=group_col,
        value_col="cumulative_regret",
    )

    agg = downsample_rounds(agg, stride=stride)

    plt.figure(figsize=(8, 5))

    for name, sub in agg.groupby(group_col):
        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        label = f"m={name}" if group_col == "feedback_m" else str(name)

        plt.plot(x, y, label=label)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel("Cumulative top-k regret")
    #plt.title("Cumulative regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_instantaneous_regret(
    history_df: pd.DataFrame,
    *,
    group_col: str,
    output_path: Path,
    window: int,
    stride: int
) -> None:
    df = add_moving_average(
        history_df,
        group_col=group_col,
        value_col="instantaneous_regret",
        output_col="instantaneous_regret_ma",
        window=window,
    )

    agg = aggregate_curve(
        df,
        group_col=group_col,
        value_col="instantaneous_regret_ma",
    )

    agg = downsample_rounds(agg, stride=stride)

    plt.figure(figsize=(8, 5))

    for name, sub in agg.groupby(group_col):
        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        label = f"m={name}" if group_col == "feedback_m" else str(name)

        plt.plot(x, y, label=label)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel(f"Instantaneous top-k regret, moving average window={window}")
    plt.title("Instantaneous regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_final_regret_boxplot(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    output_path: Path,
) -> None:
    groups = list(summary_df[group_col].unique())

    data = [
        summary_df.loc[
            summary_df[group_col] == group,
            "final_cumulative_regret",
        ].to_numpy()
        for group in groups
    ]

    labels = [f"m={g}" if group_col == "feedback_m" else str(g) for g in groups]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("Final cumulative top-k regret")
    plt.title("Final regret across seeds")
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_success(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    output_path: Path,
) -> None:
    grouped = (
        summary_df.groupby(group_col)
        .agg(
            correctness_rate=("correct", "mean"),
            completion_rate=("completed", "mean"),
        )
        .reset_index()
    )

    x = np.arange(len(grouped))
    width = 0.35

    labels = [
        f"m={g}" if group_col == "feedback_m" else str(g)
        for g in grouped[group_col]
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, grouped["correctness_rate"], width, label="correct")
    plt.bar(x + width / 2, grouped["completion_rate"], width, label="completed")

    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Fraction of seeds")
    plt.title("Success rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_final_regret_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    grouped = (
        summary_df.groupby("feedback_m")["final_cumulative_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    x = grouped["feedback_m"].to_numpy()
    y = grouped["mean"].to_numpy()
    err = grouped["ci95"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=err, marker="o", capsize=4)
    plt.xlabel("Feedback level m")
    plt.ylabel("Final cumulative top-k regret")
    plt.title("Final regret vs feedback amount")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_completion_round_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    if "completion_round" not in summary_df.columns:
        return

    grouped = (
        summary_df.groupby("feedback_m")["completion_round"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    x = grouped["feedback_m"].to_numpy()
    y = grouped["mean"].to_numpy()
    err = grouped["ci95"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=err, marker="o", capsize=4)
    plt.xlabel("Feedback level m")
    plt.ylabel("Mean completion round")
    plt.title("Completion round vs feedback amount")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing history.csv and summary.csv.",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Algorithms to exclude from plots, e.g. --exclude uniform.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Moving-average window for instantaneous regret.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Subsample of rounds for manageable PDFs.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    history_path = results_dir / "history.csv"
    summary_path = results_dir / "summary.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    history_df = pd.read_csv(history_path)
    summary_df = pd.read_csv(summary_path)

    if args.exclude:
        history_df = history_df[~history_df["algorithm"].isin(args.exclude)].copy()
        summary_df = summary_df[~summary_df["algorithm"].isin(args.exclude)].copy()

    pdf_dir = results_dir / "pdf_plots"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Feedback-sensitivity runs have feedback_m.
    if "feedback_m" in history_df.columns:
        group_col = "feedback_m"

        history_df["feedback_m"] = history_df["feedback_m"].astype(int)
        summary_df["feedback_m"] = summary_df["feedback_m"].astype(int)

        cumulative_name = "cumulative_regret_by_m.pdf"
        instantaneous_name = "instantaneous_regret_by_m.pdf"
        final_box_name = "final_regret_boxplot_by_m.pdf"
        success_name = "success_by_m.pdf"

    else:
        group_col = "algorithm"

        cumulative_name = "cumulative_regret.pdf"
        instantaneous_name = "instantaneous_regret.pdf"
        final_box_name = "final_regret_boxplot.pdf"
        success_name = "success.pdf"

    num_saved_points = history_df["round"].nunique()

    if args.window is None:
        window = max(1, min(500, num_saved_points // 30))
    else:
        window = args.window

    plot_cumulative_regret(
        history_df,
        group_col=group_col,
        output_path=pdf_dir / cumulative_name,
        stride=args.stride
    )

    plot_instantaneous_regret(
        history_df,
        group_col=group_col,
        output_path=pdf_dir / instantaneous_name,
        window=window,
        stride=args.stride
    )

    plot_final_regret_boxplot(
        summary_df,
        group_col=group_col,
        output_path=pdf_dir / final_box_name,
    )

    plot_success(
        summary_df,
        group_col=group_col,
        output_path=pdf_dir / success_name,
    )

    if group_col == "feedback_m":
        plot_final_regret_vs_m(
            summary_df,
            output_path=pdf_dir / "final_regret_vs_m.pdf",
        )

        plot_completion_round_vs_m(
            summary_df,
            output_path=pdf_dir / "completion_round_vs_m.pdf",
        )

    print(f"Saved PDF plots to: {pdf_dir}")

def downsample_rounds(df: pd.DataFrame, *, stride: int | None) -> pd.DataFrame:
    if stride is None or stride <= 1:
        return df.copy()

    min_round = int(df["round"].min())
    max_round = int(df["round"].max())

    mask = (
        (df["round"] == min_round)
        | (df["round"] == max_round)
        | (df["round"] % stride == 0)
    )

    return df.loc[mask].copy()


if __name__ == "__main__":
    main()