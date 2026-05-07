# mnl_make_plots.py

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

# General output
OUTPUT_SUBDIR = "pdf_plots"
OUTPUT_FORMAT = "pdf"
DPI = 300

# Titles
SHOW_TITLES = False

# Figure shape.
# Single plots are 4:3. A row of 3 plots will be 3 * width by height.
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

# Use real LaTeX for \textsc.
# Set to False if LaTeX is not installed.
USE_TEX = True

# Axis labels
X_LABEL_HORIZON = r"Learning Horizon $T$"
X_LABEL_THETA0 = r'"No-Choice" Parameter $\theta_0$'
Y_LABEL_REGRET = r"Cumulative Top-$k$ Regret"

# Axis limits
HORIZON_X_MIN = -0.1
HORIZON_X_MAX = None        # None means infer max round from history.csv
HORIZON_X_SCALE = "linear"
REGRET_Y_MIN = -10000
REGRET_Y_MAX = None         # None means automatic
REGRET_Y_SCALE = "log"   # or "log"

THETA0_X_MIN = -0.1
THETA0_X_MAX = 1.1
THETA0_X_SCALE = "log"   # "linear" or "log"

# Downsampling.
# Example: if T = 1_000_000 and PLOT_STRIDE = 1000,
# only points at 1000, 2000, ... are plotted, plus first/last.
PLOT_STRIDE = 1000

# Moving average window for instantaneous regret.
# If None, the script chooses one automatically.
MOVING_AVERAGE_WINDOW = None

# Algorithm display names.
# These labels work without requiring LaTeX installation.
ALGORITHM_LABELS_TEX = {
    "topk_uif": r"\textsc{Top-}$k$\textsc{-UIF}",
    "mnl_topk_uif": r"\textsc{MNL-Top-}$k$\textsc{-UIF}",
    "rec_maxmin_ucb": r"\textsc{RecMaxMinUCB}",
    "sg_no_choice": r"\textsc{AOA-RB}$_{\mathrm{PL}}$",
    "sg_adaptive": r"\textsc{AOA-RB}$_{\mathrm{PL}}$\textsc{-Adaptive}",
}

ALGORITHM_LABELS_NO_TEX = {
    "topk_uif": r"Top-$k$-UIF",
    "mnl_topk_uif": r"MNL-Top-$k$-UIF",
    "rec_maxmin_ucb": "RecMaxMinUCB",
    "sg_no_choice": r"AOA-RB$_{\mathrm{PL}}$",
    "sg_adaptive": r"AOA-RB$_{\mathrm{PL}}$-Adaptive",
}

# Optional line styles.
ALGORITHM_LINESTYLES = {
    "topk_uif": "-",
    "mnl_topk_uif": "-",
    "rec_maxmin_ucb": "--",
    "sg_no_choice": "--",
    "sg_adaptive": "-.",
}

# Optional fixed ordering in legends.
ALGORITHM_ORDER = [
    "topk_uif",
    "mnl_topk_uif",
    "rec_maxmin_ucb",
    "sg_adaptive",
    "sg_no_choice",
]

# Produce a single row with 3 cumulative-regret plots.
MAKE_THREE_PANEL_ROW = True

# If None, the script uses the first 3 theta0 values available.
# Otherwise set, for example: ROW_THETA0_VALUES = [1.0, 0.1, 0.01]
ROW_THETA0_VALUES = None

# Save individual per-theta0 plots too.
SAVE_INDIVIDUAL_THETA0_PLOTS = True

# Save instantaneous-regret plots too.
SAVE_INSTANTANEOUS_PLOTS = True

# Save summary-vs-theta0 plots.
SAVE_THETA0_SUMMARY_PLOTS = True


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

def algorithm_label(name: str) -> str:
    if USE_TEX:
        return ALGORITHM_LABELS_TEX.get(name, name)
    return ALGORITHM_LABELS_NO_TEX.get(name, name)


def algorithm_linestyle(name: str) -> str:
    return ALGORITHM_LINESTYLES.get(name, "-")


def sorted_algorithms(names) -> list[str]:
    names = list(names)
    rank = {name: i for i, name in enumerate(ALGORITHM_ORDER)}
    return sorted(names, key=lambda x: rank.get(x, 10_000))


def setup_axis(
    ax,
    *,
    xlabel: str,
    ylabel: str,
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlabel == X_LABEL_HORIZON:
        ax.set_xscale(HORIZON_X_SCALE)

    if xlabel == X_LABEL_THETA0:
        ax.set_xscale(THETA0_X_SCALE)

    ax.set_yscale(REGRET_Y_SCALE)

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    if SHOW_GRID:
        ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)


def maybe_title(ax, title: str) -> None:
    if SHOW_TITLES:
        ax.set_title(title)


def savefig(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, format=OUTPUT_FORMAT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


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

    required_history = {
        "algorithm",
        "seed",
        "round",
        "instantaneous_regret",
        "cumulative_regret",
    }
    missing = required_history - set(history_df.columns)
    if missing:
        raise ValueError(f"history.csv is missing columns: {sorted(missing)}")

    required_summary = {
        "algorithm",
        "seed",
        "final_cumulative_regret",
    }
    missing = required_summary - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary.csv is missing columns: {sorted(missing)}")

    if "theta0" in history_df.columns:
        history_df["theta0"] = history_df["theta0"].astype(float)
    if "theta0" in summary_df.columns:
        summary_df["theta0"] = summary_df["theta0"].astype(float)

    return history_df, summary_df


def filter_data(
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    algorithms: Optional[list[str]],
    exclude: Optional[list[str]],
    theta0_values: Optional[list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = history_df.copy()
    s = summary_df.copy()

    if algorithms:
        h = h[h["algorithm"].isin(algorithms)].copy()
        s = s[s["algorithm"].isin(algorithms)].copy()

    if exclude:
        h = h[~h["algorithm"].isin(exclude)].copy()
        s = s[~s["algorithm"].isin(exclude)].copy()

    if theta0_values and "theta0" in h.columns:
        theta0_values = [float(x) for x in theta0_values]
        h = h[h["theta0"].isin(theta0_values)].copy()
        s = s[s["theta0"].isin(theta0_values)].copy()

    if len(h) == 0:
        raise ValueError("No history rows left after filtering.")
    if len(s) == 0:
        raise ValueError("No summary rows left after filtering.")

    return h, s


# ============================================================
# Aggregation
# ============================================================

def grouping_columns(history_df: pd.DataFrame) -> list[str]:
    if "theta0" in history_df.columns:
        return ["theta0", "algorithm", "round"]
    return ["algorithm", "round"]


def aggregate_curve(
    history_df: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    group_cols = grouping_columns(history_df)

    grouped = (
        history_df
        .groupby(group_cols)[value_col]
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
    sort_cols = ["algorithm", "seed", "round"]
    group_cols = ["algorithm", "seed"]

    if "theta0" in history_df.columns:
        sort_cols = ["theta0"] + sort_cols
        group_cols = ["theta0"] + group_cols

    df = history_df.sort_values(sort_cols).copy()

    df[output_col] = (
        df
        .groupby(group_cols)[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def downsample_rounds(
    df: pd.DataFrame,
    *,
    stride: Optional[int],
) -> pd.DataFrame:
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


# ============================================================
# Curve plotting
# ============================================================

def plot_one_curve_panel(
    ax,
    curve_df: pd.DataFrame,
    *,
    x_max: int,
    title: str,
    show_legend: bool,
) -> None:
    algorithms = sorted_algorithms(curve_df["algorithm"].unique())

    for algorithm_name in algorithms:
        sub = curve_df[curve_df["algorithm"] == algorithm_name].sort_values("round")

        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        ax.plot(
            x,
            y,
            label=algorithm_label(algorithm_name),
            linewidth=LINE_WIDTH,
            linestyle=algorithm_linestyle(algorithm_name),
        )

        if SHOW_CI:
            ax.fill_between(x, y - ci, y + ci, alpha=CI_ALPHA)

    setup_axis(
        ax,
        xlabel=X_LABEL_HORIZON,
        ylabel=Y_LABEL_REGRET,
        x_min=HORIZON_X_MIN,
        x_max=x_max,
        y_min=REGRET_Y_MIN,
        y_max=REGRET_Y_MAX,
    )

    maybe_title(ax, title)

    if show_legend:
        ax.legend(frameon=True)


def plot_cumulative_regret(
    history_df: pd.DataFrame,
    *,
    output_dir: Path,
    stride: Optional[int],
) -> None:
    agg = aggregate_curve(history_df, value_col="cumulative_regret")
    agg = downsample_rounds(agg, stride=stride)

    x_max = infer_horizon_xmax(history_df)

    if "theta0" in agg.columns:
        if SAVE_INDIVIDUAL_THETA0_PLOTS:
            for theta0, sub in agg.groupby("theta0"):
                fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
                plot_one_curve_panel(
                    ax,
                    sub,
                    x_max=x_max,
                    title=fr"Expected Top-$k$ Regret, $\theta_0={theta0:g}$",
                    show_legend=True,
                )
                savefig(fig, output_dir / f"cumulative_regret_theta0_{theta0:g}.pdf")
    else:
        fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
        plot_one_curve_panel(
            ax,
            agg,
            x_max=x_max,
            title=r"Expected Top-$k$ Regret",
            show_legend=True,
        )
        savefig(fig, output_dir / "cumulative_regret.pdf")


def plot_instantaneous_regret(
    history_df: pd.DataFrame,
    *,
    output_dir: Path,
    window: int,
    stride: Optional[int],
) -> None:
    df = add_moving_average(
        history_df,
        value_col="instantaneous_regret",
        output_col="instantaneous_regret_ma",
        window=window,
    )

    agg = aggregate_curve(df, value_col="instantaneous_regret_ma")
    agg = downsample_rounds(agg, stride=stride)

    x_max = infer_horizon_xmax(history_df)

    if "theta0" in agg.columns:
        if SAVE_INDIVIDUAL_THETA0_PLOTS:
            for theta0, sub in agg.groupby("theta0"):
                fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
                plot_one_curve_panel(
                    ax,
                    sub,
                    x_max=x_max,
                    title=fr"Instantaneous Regret, $\theta_0={theta0:g}$",
                    show_legend=True,
                )
                savefig(fig, output_dir / f"instantaneous_regret_theta0_{theta0:g}.pdf")
    else:
        fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
        plot_one_curve_panel(
            ax,
            agg,
            x_max=x_max,
            title="Instantaneous Regret",
            show_legend=True,
        )
        savefig(fig, output_dir / "instantaneous_regret.pdf")


def choose_row_theta0_values(history_df: pd.DataFrame) -> list[float]:
    available = sorted(history_df["theta0"].unique(), reverse=True)

    if ROW_THETA0_VALUES is not None:
        chosen = [float(x) for x in ROW_THETA0_VALUES]
        missing = sorted(set(chosen) - set(available))
        if missing:
            raise ValueError(f"Requested ROW_THETA0_VALUES not found: {missing}")
        return chosen

    if len(available) < 3:
        return available

    return available[:3]


def plot_three_panel_cumulative_row(
    history_df: pd.DataFrame,
    *,
    output_dir: Path,
    stride: Optional[int],
) -> None:
    if "theta0" not in history_df.columns:
        return

    theta0_values = choose_row_theta0_values(history_df)

    if len(theta0_values) == 0:
        return

    agg = aggregate_curve(history_df, value_col="cumulative_regret")
    agg = downsample_rounds(agg, stride=stride)

    x_max = infer_horizon_xmax(history_df)

    fig, axes = plt.subplots(
        1,
        len(theta0_values),
        figsize=(FIGWIDTH * len(theta0_values), FIGHEIGHT),
        sharey=True,
    )

    if len(theta0_values) == 1:
        axes = [axes]

    for ax, theta0 in zip(axes, theta0_values):
        sub = agg[agg["theta0"] == theta0]

        plot_one_curve_panel(
            ax,
            sub,
            x_max=x_max,
            title=fr"$\theta_0={theta0:g}$",
            show_legend=False,
        )

        # In a row, only the first panel needs the y-label.
        if ax is not axes[0]:
            ax.set_ylabel("")

        # With SHOW_TITLES=False, put theta0 in a small annotation instead.
        if not SHOW_TITLES:
            ax.text(
                0.04,
                0.94,
                fr"$\theta_0={theta0:g}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=AXIS_LABEL_SIZE,
            )

    handles, labels = axes[0].get_legend_handles_labels()
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
        output_dir / "cumulative_regret_three_panel_row.pdf",
        format=OUTPUT_FORMAT,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Theta0 summary plots
# ============================================================

def aggregate_summary_metric(
    summary_df: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    grouped = (
        summary_df
        .groupby(["theta0", "algorithm"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    return grouped


def plot_metric_vs_theta0(
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    metric: str,
    output_name: str,
    y_label: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    if "theta0" not in summary_df.columns:
        return

    if metric not in summary_df.columns:
        print(f"Skipping {output_name}: missing column {metric}.")
        return

    grouped = aggregate_summary_metric(summary_df, metric=metric)

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

    algorithms = sorted_algorithms(grouped["algorithm"].unique())

    for algorithm_name in algorithms:
        sub = grouped[grouped["algorithm"] == algorithm_name].sort_values("theta0")

        x = sub["theta0"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        ax.errorbar(
            x,
            y,
            yerr=ci if SHOW_CI else None,
            marker="o",
            capsize=4,
            linewidth=LINE_WIDTH,
            linestyle=algorithm_linestyle(algorithm_name),
            label=algorithm_label(algorithm_name),
        )

    if THETA0_X_SCALE == "log":
        ax.set_xscale("log")
        x_min = max(min(grouped["theta0"]), 1e-12)
    else:
        x_min = THETA0_X_MIN

    setup_axis(
        ax,
        xlabel=X_LABEL_THETA0,
        ylabel=y_label,
        x_min=x_min,
        x_max=THETA0_X_MAX,
        y_min=y_min,
        y_max=y_max,
    )

    maybe_title(ax, y_label)
    ax.legend(frameon=True)

    savefig(fig, output_dir / output_name)


def plot_theta0_summaries(summary_df: pd.DataFrame, *, output_dir: Path) -> None:
    if "theta0" not in summary_df.columns:
        return

    plot_metric_vs_theta0(
        summary_df,
        output_dir=output_dir,
        metric="final_cumulative_regret",
        output_name="final_regret_vs_theta0.pdf",
        y_label=Y_LABEL_REGRET,
        y_min=REGRET_Y_MIN,
        y_max=REGRET_Y_MAX,
    )

    plot_metric_vs_theta0(
        summary_df,
        output_dir=output_dir,
        metric="tail_mean_instantaneous_regret",
        output_name="tail_instantaneous_regret_vs_theta0.pdf",
        y_label=Y_LABEL_REGRET,
        y_min=REGRET_Y_MIN,
        y_max=REGRET_Y_MAX,
    )

    plot_metric_vs_theta0(
        summary_df,
        output_dir=output_dir,
        metric="last_action_correct",
        output_name="last_action_success_vs_theta0.pdf",
        y_label="Exact Top-$k$ Selection Rate",
        y_min=0,
        y_max=1.05,
    )


# ============================================================
# No-choice diagnostic
# ============================================================

def plot_no_choice_rate(
    history_df: pd.DataFrame,
    *,
    output_dir: Path,
    window: int,
    stride: Optional[int],
) -> None:
    if "theta0" not in history_df.columns:
        return

    if "no_choice_selected" not in history_df.columns:
        return

    df = add_moving_average(
        history_df,
        value_col="no_choice_selected",
        output_col="no_choice_rate_ma",
        window=window,
    )

    agg = aggregate_curve(df, value_col="no_choice_rate_ma")
    agg = downsample_rounds(agg, stride=stride)

    x_max = infer_horizon_xmax(history_df)

    for theta0, sub_theta in agg.groupby("theta0"):
        fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

        algorithms = sorted_algorithms(sub_theta["algorithm"].unique())

        for algorithm_name in algorithms:
            sub = sub_theta[sub_theta["algorithm"] == algorithm_name].sort_values("round")

            x = sub["round"].to_numpy()
            y = sub["mean"].to_numpy()
            ci = sub["ci95"].to_numpy()

            ax.plot(
                x,
                y,
                label=algorithm_label(algorithm_name),
                linewidth=LINE_WIDTH,
                linestyle=algorithm_linestyle(algorithm_name),
            )

            if SHOW_CI:
                ax.fill_between(x, y - ci, y + ci, alpha=CI_ALPHA)

        setup_axis(
            ax,
            xlabel=X_LABEL_HORIZON,
            ylabel="No-Choice Selection Rate",
            x_min=HORIZON_X_MIN,
            x_max=x_max,
            y_min=0,
            y_max=1.0,
        )

        maybe_title(ax, fr"No-Choice Selection Rate, $\theta_0={theta0:g}$")
        ax.legend(frameon=True)

        savefig(fig, output_dir / f"no_choice_rate_theta0_{theta0:g}.pdf")


# ============================================================
# Aggregate summary
# ============================================================

def write_aggregate_summary(summary_df: pd.DataFrame, *, output_dir: Path) -> None:
    group_cols = ["algorithm"]
    if "theta0" in summary_df.columns:
        group_cols = ["theta0", "algorithm"]

    metrics = {
        "mean_final_regret": ("final_cumulative_regret", "mean"),
        "std_final_regret": ("final_cumulative_regret", "std"),
    }

    optional = {
        "last_action_success": "last_action_correct",
        "mean_tail_instantaneous_regret": "tail_mean_instantaneous_regret",
        "recommended_success": "recommended_correct",
        "completion_rate": "completed",
    }

    for output_name, col in optional.items():
        if col in summary_df.columns:
            metrics[output_name] = (col, "mean")

    agg = summary_df.groupby(group_cols).agg(**metrics).reset_index()
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
        "--theta0-values",
        nargs="*",
        type=float,
        default=None,
        help="Only plot these theta0 values.",
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
        theta0_values=args.theta0_values,
    )

    num_saved_points = history_df["round"].nunique()

    if args.window is None:
        window = max(1, min(500, num_saved_points // 30))
    else:
        window = int(args.window)

    plot_cumulative_regret(
        history_df,
        output_dir=output_dir,
        stride=args.stride,
    )

    if SAVE_INSTANTANEOUS_PLOTS:
        plot_instantaneous_regret(
            history_df,
            output_dir=output_dir,
            window=window,
            stride=args.stride,
        )

    if MAKE_THREE_PANEL_ROW:
        plot_three_panel_cumulative_row(
            history_df,
            output_dir=output_dir,
            stride=args.stride,
        )

    if SAVE_THETA0_SUMMARY_PLOTS:
        plot_theta0_summaries(summary_df, output_dir=output_dir)

    plot_no_choice_rate(
        history_df,
        output_dir=output_dir,
        window=window,
        stride=args.stride,
    )

    write_aggregate_summary(summary_df, output_dir=output_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()