# make_pdf_plots.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PAPER PLOT CONFIGURATION
# Edit this block only
# ============================================================

OUTPUT_SUBDIR = "pdf_plots"
OUTPUT_FORMAT = "pdf"
DPI = 300

# Use LaTeX for \textsc labels.
# If this crashes because LaTeX is not installed, set USE_TEX = False.
USE_TEX = True

# Titles
SHOW_TITLES = False

# Figure shape: single plots are 4:3.
# A row of 3 plots will be 3 * width by height.
FIGWIDTH = 4.8
FIGHEIGHT = 3.6

# Font sizes, chosen to remain readable when 3 plots are placed in one row.
FONT_SIZE = 12
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 10

# Lines and confidence bands
LINE_WIDTH = 2.0
CI_ALPHA = 0.18
SHOW_CI = True

# Grid
SHOW_GRID = True
GRID_COLOR = "0.85"
GRID_LINEWIDTH = 0.8
GRID_ALPHA = 1.0

# Axis labels
X_LABEL_HORIZON = "Learning Horizon"
X_LABEL_FEEDBACK = "Feedback Level $m$"
Y_LABEL_REGRET = r"Cumulative Top-$k$ Regret"
Y_LABEL_SUCCESS = r"Exact Top-$k$ Selection Rate"
Y_LABEL_COMPLETION = "Completion Round"

# Axis scales
HORIZON_X_SCALE = "linear"   # "linear" or "log"
FEEDBACK_X_SCALE = "linear"  # usually "linear"
REGRET_Y_SCALE = "linear"    # "linear" or "log"

# Axis limits
# For log scale, minima must be positive.
HORIZON_X_MIN = 0
HORIZON_X_MAX = None         # None means infer max round from history.csv

FEEDBACK_X_MIN = None
FEEDBACK_X_MAX = None

REGRET_Y_MIN = 0
REGRET_Y_MAX = None

SUCCESS_Y_MIN = 0
SUCCESS_Y_MAX = 1.05

# Downsampling.
# Example: if T = 1_000_000 and PLOT_STRIDE = 1000,
# only points at 1000, 2000, ... are plotted, plus first/last.
PLOT_STRIDE = 1000

# Moving average window for instantaneous regret.
# If None, the script chooses one automatically.
MOVING_AVERAGE_WINDOW = None

# Algorithm display names.
# If USE_TEX=True, these use \textsc.
ALGORITHM_LABELS_TEX = {
    "topk_uif": r"\textsc{Top-$k$-UIF}",
    "rec_maxmin_ucb": r"\textsc{RecMaxMinUCB}",
    "uniform": r"\textsc{Uniform}",
}

ALGORITHM_LABELS_NO_TEX = {
    "topk_uif": r"Top-$k$-UIF",
    "rec_maxmin_ucb": "RecMaxMinUCB",
    "uniform": "Uniform",
}

# Optional line styles.
ALGORITHM_LINESTYLES = {
    "topk_uif": "-",
    "rec_maxmin_ucb": "--",
    "uniform": ":",
}

# Optional fixed ordering in legends.
ALGORITHM_ORDER = [
    "topk_uif",
    "rec_maxmin_ucb",
    "uniform",
]

# Feedback-sensitivity label format.
# Used when the CSV contains column feedback_m.
FEEDBACK_LABEL_TEMPLATE = r"$m={m}$"

# Produce a single row with up to 3 plots.
# For feedback-sensitivity results, this creates:
#   cumulative regret, instantaneous regret, final regret vs m
MAKE_THREE_PANEL_ROW = True

# Save individual plots too.
SAVE_INDIVIDUAL_PLOTS = True

# Save instantaneous-regret plots.
SAVE_INSTANTANEOUS_PLOTS = True

# Save success/completion plots.
SAVE_SUCCESS_PLOTS = True


# ============================================================
# Matplotlib style
# ============================================================

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "text.usetex": USE_TEX,
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.titlesize": AXIS_LABEL_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def group_label(group_col: str, value) -> str:
    if group_col == "feedback_m":
        return FEEDBACK_LABEL_TEMPLATE.format(m=int(value))

    if group_col == "algorithm":
        if USE_TEX:
            return ALGORITHM_LABELS_TEX.get(str(value), str(value))
        return ALGORITHM_LABELS_NO_TEX.get(str(value), str(value))

    return str(value)


def group_linestyle(group_col: str, value) -> str:
    if group_col == "algorithm":
        return ALGORITHM_LINESTYLES.get(str(value), "-")
    return "-"


def sorted_groups(group_col: str, values) -> list:
    values = list(values)

    if group_col == "algorithm":
        rank = {name: i for i, name in enumerate(ALGORITHM_ORDER)}
        return sorted(values, key=lambda x: rank.get(str(x), 10_000))

    if group_col == "feedback_m":
        return sorted(values)

    return sorted(values)


def maybe_title(ax, title: str) -> None:
    if SHOW_TITLES:
        ax.set_title(title)


def savefig(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, format=OUTPUT_FORMAT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def set_axis_scales_and_limits(
    ax,
    *,
    xlabel: str,
    ylabel: str,
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    x_scale: str,
    y_scale: str,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    # Log axes cannot include zero.
    if x_scale == "log" and x_min is not None and x_min <= 0:
        x_min = 1
    if y_scale == "log" and y_min is not None and y_min <= 0:
        y_min = 1e-8

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    if SHOW_GRID:
        ax.grid(
            True,
            color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
        ax.set_axisbelow(True)


# ============================================================
# Loading and filtering
# ============================================================

def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_path = results_dir / "history.csv"
    summary_path = results_dir / "summary.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    history_df = pd.read_csv(history_path)
    summary_df = pd.read_csv(summary_path)

    required_history_cols = {
        "algorithm",
        "seed",
        "round",
        "instantaneous_regret",
        "cumulative_regret",
    }

    missing = required_history_cols - set(history_df.columns)
    if missing:
        raise ValueError(f"history.csv is missing columns: {sorted(missing)}")

    required_summary_cols = {
        "algorithm",
        "seed",
        "final_cumulative_regret",
    }

    missing = required_summary_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary.csv is missing columns: {sorted(missing)}")

    if "feedback_m" in history_df.columns:
        history_df["feedback_m"] = history_df["feedback_m"].astype(int)

    if "feedback_m" in summary_df.columns:
        summary_df["feedback_m"] = summary_df["feedback_m"].astype(int)

    return history_df, summary_df


def filter_data(
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    algorithms: Optional[list[str]],
    exclude: Optional[list[str]],
    feedback_values: Optional[list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = history_df.copy()
    s = summary_df.copy()

    if algorithms:
        h = h[h["algorithm"].isin(algorithms)].copy()
        s = s[s["algorithm"].isin(algorithms)].copy()

    if exclude:
        h = h[~h["algorithm"].isin(exclude)].copy()
        s = s[~s["algorithm"].isin(exclude)].copy()

    if feedback_values and "feedback_m" in h.columns:
        feedback_values = [int(x) for x in feedback_values]
        h = h[h["feedback_m"].isin(feedback_values)].copy()
        s = s[s["feedback_m"].isin(feedback_values)].copy()

    if len(h) == 0:
        raise ValueError("No history rows left after filtering.")
    if len(s) == 0:
        raise ValueError("No summary rows left after filtering.")

    return h, s


def infer_group_col(history_df: pd.DataFrame) -> str:
    if "feedback_m" in history_df.columns:
        return "feedback_m"
    return "algorithm"


# ============================================================
# Aggregation
# ============================================================

def aggregate_curve(
    history_df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    grouped = (
        history_df
        .groupby([group_col, "round"])[value_col]
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
    group_col: str,
    value_col: str,
    output_col: str,
    window: int,
) -> pd.DataFrame:
    df = history_df.sort_values([group_col, "seed", "round"]).copy()

    df[output_col] = (
        df
        .groupby([group_col, "seed"])[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def downsample_rounds(
    df: pd.DataFrame,
    *,
    stride: Optional[int],
) -> pd.DataFrame:
    """
    Downsample after aggregation/smoothing.
    Keeps first round, last round, and rounds divisible by stride.
    """
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


def infer_horizon_xmax(history_df: pd.DataFrame) -> int:
    if HORIZON_X_MAX is not None:
        return int(HORIZON_X_MAX)
    return int(history_df["round"].max())


def safe_curve_values(y: np.ndarray, ci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Avoid invalid values when using log-scale y.
    """
    if REGRET_Y_SCALE != "log":
        return y, y - ci, y + ci

    floor = REGRET_Y_MIN if REGRET_Y_MIN and REGRET_Y_MIN > 0 else 1e-8

    y_safe = np.maximum(y, floor)
    lower = np.maximum(y - ci, floor)
    upper = np.maximum(y + ci, floor)

    return y_safe, lower, upper


# ============================================================
# Main curve plots
# ============================================================

def plot_curve_panel(
    ax,
    curve_df: pd.DataFrame,
    *,
    group_col: str,
    x_max: int,
    title: str,
    show_legend: bool,
) -> None:
    groups = sorted_groups(group_col, curve_df[group_col].unique())

    for group in groups:
        sub = curve_df[curve_df[group_col] == group].sort_values("round")

        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        y_plot, lower, upper = safe_curve_values(y, ci)

        ax.plot(
            x,
            y_plot,
            label=group_label(group_col, group),
            linewidth=LINE_WIDTH,
            linestyle=group_linestyle(group_col, group),
        )

        if SHOW_CI:
            ax.fill_between(x, lower, upper, alpha=CI_ALPHA)

    set_axis_scales_and_limits(
        ax,
        xlabel=X_LABEL_HORIZON,
        ylabel=Y_LABEL_REGRET,
        x_min=HORIZON_X_MIN,
        x_max=x_max,
        y_min=REGRET_Y_MIN,
        y_max=REGRET_Y_MAX,
        x_scale=HORIZON_X_SCALE,
        y_scale=REGRET_Y_SCALE,
    )

    maybe_title(ax, title)

    if show_legend:
        ax.legend(frameon=True)


def plot_cumulative_regret(
    history_df: pd.DataFrame,
    *,
    group_col: str,
    output_dir: Path,
    stride: Optional[int],
) -> None:
    agg = aggregate_curve(
        history_df,
        group_col=group_col,
        value_col="cumulative_regret",
    )
    agg = downsample_rounds(agg, stride=stride)

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    plot_curve_panel(
        ax,
        agg,
        group_col=group_col,
        x_max=infer_horizon_xmax(history_df),
        title="Cumulative Regret",
        show_legend=True,
    )

    output_name = (
        "cumulative_regret_by_m.pdf"
        if group_col == "feedback_m"
        else "cumulative_regret.pdf"
    )

    savefig(fig, output_dir / output_name)


def plot_instantaneous_regret(
    history_df: pd.DataFrame,
    *,
    group_col: str,
    output_dir: Path,
    window: int,
    stride: Optional[int],
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

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    plot_curve_panel(
        ax,
        agg,
        group_col=group_col,
        x_max=infer_horizon_xmax(history_df),
        title="Instantaneous Regret",
        show_legend=True,
    )

    output_name = (
        "instantaneous_regret_by_m.pdf"
        if group_col == "feedback_m"
        else "instantaneous_regret.pdf"
    )

    savefig(fig, output_dir / output_name)


# ============================================================
# Summary plots
# ============================================================

def aggregate_summary_metric(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    metric: str,
) -> pd.DataFrame:
    grouped = (
        summary_df
        .groupby(group_col)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    return grouped


def plot_final_regret_boxplot(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    output_dir: Path,
) -> None:
    groups = sorted_groups(group_col, summary_df[group_col].unique())

    data = [
        summary_df.loc[
            summary_df[group_col] == group,
            "final_cumulative_regret",
        ].to_numpy()
        for group in groups
    ]

    labels = [group_label(group_col, group) for group in groups]

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    ax.boxplot(data, labels=labels)

    set_axis_scales_and_limits(
        ax,
        xlabel=X_LABEL_FEEDBACK if group_col == "feedback_m" else "",
        ylabel=Y_LABEL_REGRET,
        x_min=None,
        x_max=None,
        y_min=REGRET_Y_MIN,
        y_max=REGRET_Y_MAX,
        x_scale="linear",
        y_scale=REGRET_Y_SCALE,
    )

    maybe_title(ax, "Final Regret")
    savefig(fig, output_dir / "final_regret_boxplot.pdf")


def plot_success(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    output_dir: Path,
) -> None:
    if "correct" not in summary_df.columns or "completed" not in summary_df.columns:
        print("Skipping success plot: missing correct/completed columns.")
        return

    grouped = (
        summary_df
        .groupby(group_col)
        .agg(
            correctness_rate=("correct", "mean"),
            completion_rate=("completed", "mean"),
        )
        .reset_index()
    )

    groups = sorted_groups(group_col, grouped[group_col].unique())
    grouped = grouped.set_index(group_col).loc[groups].reset_index()

    x = np.arange(len(grouped))
    width = 0.35

    labels = [group_label(group_col, group) for group in grouped[group_col]]

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    ax.bar(
        x - width / 2,
        grouped["correctness_rate"],
        width,
        label="Correct",
    )
    ax.bar(
        x + width / 2,
        grouped["completion_rate"],
        width,
        label="Completed",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    set_axis_scales_and_limits(
        ax,
        xlabel=X_LABEL_FEEDBACK if group_col == "feedback_m" else "",
        ylabel=Y_LABEL_SUCCESS,
        x_min=None,
        x_max=None,
        y_min=SUCCESS_Y_MIN,
        y_max=SUCCESS_Y_MAX,
        x_scale="linear",
        y_scale="linear",
    )

    maybe_title(ax, "Success Rates")
    ax.legend(frameon=True)

    output_name = "success_by_m.pdf" if group_col == "feedback_m" else "success.pdf"
    savefig(fig, output_dir / output_name)


def plot_final_regret_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    if "feedback_m" not in summary_df.columns:
        return

    grouped = aggregate_summary_metric(
        summary_df,
        group_col="feedback_m",
        metric="final_cumulative_regret",
    ).sort_values("feedback_m")

    x = grouped["feedback_m"].to_numpy()
    y = grouped["mean"].to_numpy()
    ci = grouped["ci95"].to_numpy()

    y_plot, lower, upper = safe_curve_values(y, ci)

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    ax.errorbar(
        x,
        y_plot,
        yerr=(y_plot - lower, upper - y_plot) if SHOW_CI else None,
        marker="o",
        capsize=4,
        linewidth=LINE_WIDTH,
    )

    set_axis_scales_and_limits(
        ax,
        xlabel=X_LABEL_FEEDBACK,
        ylabel=Y_LABEL_REGRET,
        x_min=FEEDBACK_X_MIN,
        x_max=FEEDBACK_X_MAX,
        y_min=REGRET_Y_MIN,
        y_max=REGRET_Y_MAX,
        x_scale=FEEDBACK_X_SCALE,
        y_scale=REGRET_Y_SCALE,
    )

    ax.set_xticks(x)
    maybe_title(ax, "Final Regret vs Feedback")
    savefig(fig, output_dir / "final_regret_vs_m.pdf")


def plot_completion_round_vs_m(
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    if "feedback_m" not in summary_df.columns:
        return

    if "completion_round" not in summary_df.columns:
        print("Skipping completion round plot: missing completion_round.")
        return

    grouped = aggregate_summary_metric(
        summary_df,
        group_col="feedback_m",
        metric="completion_round",
    ).sort_values("feedback_m")

    x = grouped["feedback_m"].to_numpy()
    y = grouped["mean"].to_numpy()
    ci = grouped["ci95"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    ax.errorbar(
        x,
        y,
        yerr=ci if SHOW_CI else None,
        marker="o",
        capsize=4,
        linewidth=LINE_WIDTH,
    )

    set_axis_scales_and_limits(
        ax,
        xlabel=X_LABEL_FEEDBACK,
        ylabel=Y_LABEL_COMPLETION,
        x_min=FEEDBACK_X_MIN,
        x_max=FEEDBACK_X_MAX,
        y_min=0,
        y_max=None,
        x_scale=FEEDBACK_X_SCALE,
        y_scale="linear",
    )

    ax.set_xticks(x)
    maybe_title(ax, "Completion Round vs Feedback")
    savefig(fig, output_dir / "completion_round_vs_m.pdf")


# ============================================================
# Three-panel row
# ============================================================

def plot_three_panel_row(
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    output_dir: Path,
    window: int,
    stride: Optional[int],
) -> None:
    """
    Makes one row of 3 plots.

    For feedback-sensitivity:
        1. cumulative regret by m
        2. instantaneous regret by m
        3. final regret vs m

    For ordinary algorithm comparison:
        1. cumulative regret by algorithm
        2. instantaneous regret by algorithm
        3. final regret boxplot
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGWIDTH * 3, FIGHEIGHT),
    )

    x_max = infer_horizon_xmax(history_df)

    # Panel 1: cumulative regret
    cum = aggregate_curve(
        history_df,
        group_col=group_col,
        value_col="cumulative_regret",
    )
    cum = downsample_rounds(cum, stride=stride)

    plot_curve_panel(
        axes[0],
        cum,
        group_col=group_col,
        x_max=x_max,
        title="Cumulative Regret",
        show_legend=False,
    )

    # Panel 2: instantaneous regret
    inst_df = add_moving_average(
        history_df,
        group_col=group_col,
        value_col="instantaneous_regret",
        output_col="instantaneous_regret_ma",
        window=window,
    )
    inst = aggregate_curve(
        inst_df,
        group_col=group_col,
        value_col="instantaneous_regret_ma",
    )
    inst = downsample_rounds(inst, stride=stride)

    plot_curve_panel(
        axes[1],
        inst,
        group_col=group_col,
        x_max=x_max,
        title="Instantaneous Regret",
        show_legend=False,
    )

    # Panel 3
    if group_col == "feedback_m":
        grouped = aggregate_summary_metric(
            summary_df,
            group_col="feedback_m",
            metric="final_cumulative_regret",
        ).sort_values("feedback_m")

        x = grouped["feedback_m"].to_numpy()
        y = grouped["mean"].to_numpy()
        ci = grouped["ci95"].to_numpy()

        y_plot, lower, upper = safe_curve_values(y, ci)

        axes[2].errorbar(
            x,
            y_plot,
            yerr=(y_plot - lower, upper - y_plot) if SHOW_CI else None,
            marker="o",
            capsize=4,
            linewidth=LINE_WIDTH,
        )

        set_axis_scales_and_limits(
            axes[2],
            xlabel=X_LABEL_FEEDBACK,
            ylabel=Y_LABEL_REGRET,
            x_min=FEEDBACK_X_MIN,
            x_max=FEEDBACK_X_MAX,
            y_min=REGRET_Y_MIN,
            y_max=REGRET_Y_MAX,
            x_scale=FEEDBACK_X_SCALE,
            y_scale=REGRET_Y_SCALE,
        )

        axes[2].set_xticks(x)
        maybe_title(axes[2], "Final Regret")
    else:
        groups = sorted_groups(group_col, summary_df[group_col].unique())

        data = [
            summary_df.loc[
                summary_df[group_col] == group,
                "final_cumulative_regret",
            ].to_numpy()
            for group in groups
        ]
        labels = [group_label(group_col, group) for group in groups]

        axes[2].boxplot(data, labels=labels)

        set_axis_scales_and_limits(
            axes[2],
            xlabel="",
            ylabel=Y_LABEL_REGRET,
            x_min=None,
            x_max=None,
            y_min=REGRET_Y_MIN,
            y_max=REGRET_Y_MAX,
            x_scale="linear",
            y_scale=REGRET_Y_SCALE,
        )

        maybe_title(axes[2], "Final Regret")

    # Shared legend above row
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            frameon=True,
            bbox_to_anchor=(0.5, 1.08),
            fontsize=LEGEND_FONT_SIZE,
        )

    fig.tight_layout()
    fig.savefig(
        output_dir / "three_panel_row.pdf",
        format=OUTPUT_FORMAT,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Aggregate summary
# ============================================================

def write_aggregate_summary(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    output_dir: Path,
) -> None:
    metrics = {
        "mean_final_regret": ("final_cumulative_regret", "mean"),
        "std_final_regret": ("final_cumulative_regret", "std"),
    }

    optional = {
        "correctness_rate": "correct",
        "completion_rate": "completed",
        "mean_precision": "precision",
        "mean_recall": "recall",
        "mean_completion_round": "completion_round",
        "mean_tail_instantaneous_regret": "tail_mean_instantaneous_regret",
    }

    for output_name, col in optional.items():
        if col in summary_df.columns:
            metrics[output_name] = (col, "mean")

    agg = summary_df.groupby(group_col).agg(**metrics).reset_index()
    agg.to_csv(output_dir / "aggregate_summary_from_plots.csv", index=False)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing history.csv and summary.csv.",
    )

    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Only plot these algorithms.",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Exclude these algorithms.",
    )

    parser.add_argument(
        "--feedback-values",
        nargs="*",
        type=int,
        default=None,
        help="Only plot these feedback levels m.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=PLOT_STRIDE,
        help="Downsample curve plots.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=MOVING_AVERAGE_WINDOW,
        help="Moving-average window for instantaneous regret.",
    )

    return parser.parse_args()


def main() -> None:
    configure_matplotlib()

    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df, summary_df = load_results(results_dir)

    history_df, summary_df = filter_data(
        history_df,
        summary_df,
        algorithms=args.algorithms,
        exclude=args.exclude,
        feedback_values=args.feedback_values,
    )

    group_col = infer_group_col(history_df)

    num_saved_points = history_df["round"].nunique()

    if args.window is None:
        window = max(1, min(500, num_saved_points // 30))
    else:
        window = int(args.window)

    if SAVE_INDIVIDUAL_PLOTS:
        plot_cumulative_regret(
            history_df,
            group_col=group_col,
            output_dir=output_dir,
            stride=args.stride,
        )

    if SAVE_INSTANTANEOUS_PLOTS:
        plot_instantaneous_regret(
            history_df,
            group_col=group_col,
            output_dir=output_dir,
            window=window,
            stride=args.stride,
        )

    if SAVE_SUCCESS_PLOTS:
        plot_final_regret_boxplot(
            summary_df,
            group_col=group_col,
            output_dir=output_dir,
        )

        plot_success(
            summary_df,
            group_col=group_col,
            output_dir=output_dir,
        )

        plot_final_regret_vs_m(
            summary_df,
            output_dir=output_dir,
        )

        plot_completion_round_vs_m(
            summary_df,
            output_dir=output_dir,
        )

    if MAKE_THREE_PANEL_ROW:
        plot_three_panel_row(
            history_df,
            summary_df,
            group_col=group_col,
            output_dir=output_dir,
            window=window,
            stride=args.stride,
        )

    write_aggregate_summary(
        summary_df,
        group_col=group_col,
        output_dir=output_dir,
    )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()