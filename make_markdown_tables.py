# make_markdown_tables.py
#
# Turn an experiment's summary.csv into a Markdown table suitable for
# pasting directly into an OpenReview rebuttal.
#
# Two modes:
#
# 1. Simple mode (default): one row per group (e.g. per algorithm, per
#    feedback level m, per delta_min), one column per requested metric.
#
#       python make_markdown_tables.py --results-dir results/weak_kth_easier \
#           --group-col algorithm \
#           --metrics final_cumulative_regret correct completed
#
# 2. Pivot mode: a rows x cols matrix of a single metric, e.g. delta_min
#    (rows) x algorithm (cols), which is the natural shape for a hardness
#    sweep table.
#
#       python make_markdown_tables.py --results-dir results/my_hardness_sweep \
#           --pivot-metric final_cumulative_regret --rows delta_min --cols algorithm
#
# Both modes read summary.csv (one row per seed) and report mean +/- an error
# band across seeds: 95% CI by default, or +/- 1 SEM with --error sem. Rows
# can be filtered with --exclude (e.g. --exclude uniform, matched against
# --exclude-col, default "algorithm"). Run `python make_markdown_tables.py
# --help` for all options.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Display configuration
# ============================================================

ALGORITHM_LABELS = {
    "topk_uif": "Top-k-UIF",
    "rec_maxmin_ucb": "RecMaxMinUCB",
    "uniform": "Uniform",
    "mnl_topk_uif": "MNL-Top-k-UIF",
    "sg_adaptive": "AOA-RBPL-Adaptive",
    "sg_no_choice": "AOA-RBPL",
}

# Preferred left-to-right order when a group column is "algorithm".
ALGORITHM_ORDER = list(ALGORITHM_LABELS.keys())

METRIC_LABELS = {
    "final_cumulative_regret": "Final Regret",
    "correct": "Correctness Rate",
    "completed": "Completion Rate",
    "precision": "Precision",
    "recall": "Recall",
    "completion_round": "Completion Round",
    "tail_mean_instantaneous_regret": "Tail Instant. Regret",
    "tail_mean_cumulative_regret": "Tail Cumulative Regret",
}

# Metrics in [0, 1] that should be reported as percentages without a CI band.
RATE_METRICS = {"correct", "completed"}


def group_label(group_col: str, value) -> str:
    if group_col == "algorithm":
        return ALGORITHM_LABELS.get(str(value), str(value))
    if group_col == "feedback_m":
        return f"m={int(value)}"
    if group_col in {"delta_min", "boundary_gap", "top_gap"}:
        return f"{float(value):g}"
    return str(value)


def sort_key(group_col: str, value):
    if group_col == "algorithm":
        name = str(value)
        return (ALGORITHM_ORDER.index(name) if name in ALGORITHM_ORDER else 999, name)
    if group_col in {"delta_min", "boundary_gap", "top_gap"}:
        # Descending: largest (easiest) gap first, matching hardness_sweep.py's
        # and top_gap_sweep.py's left-to-right decreasing convention.
        return -float(value)
    return value


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


# ============================================================
# Loading
# ============================================================

def load_summary(results_dir: Path) -> pd.DataFrame:
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    return pd.read_csv(summary_path)


# ============================================================
# Aggregation and formatting
# ============================================================

ERROR_MULTIPLIERS = {"sem": 1.0, "ci95": 1.96}


def mean_and_error(values: pd.Series, *, error_mode: str = "ci95") -> tuple[float, float]:
    """
    error_mode: "sem" for +/- 1 standard error of the mean, or "ci95" for the
    wider +/- 1.96 * SEM 95% confidence interval (the historical default).
    """
    if error_mode not in ERROR_MULTIPLIERS:
        raise ValueError(f"error_mode must be one of {sorted(ERROR_MULTIPLIERS)}.")

    values = values.dropna()
    if len(values) == 0:
        return float("nan"), float("nan")

    mean = float(values.mean())
    std = float(values.std())
    std = 0.0 if np.isnan(std) else std
    error = ERROR_MULTIPLIERS[error_mode] * std / np.sqrt(len(values))
    return mean, error


def format_cell(
    mean: float,
    ci: Optional[float],
    *,
    precision: int,
    as_percent: bool,
) -> str:
    if np.isnan(mean):
        return "--"

    if as_percent:
        return f"{100 * mean:.{precision}f}%"

    text = f"{mean:.{precision}f}"
    if ci is not None and not np.isnan(ci):
        text += f" ± {ci:.{precision}f}"
    return text


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ============================================================
# Table builders
# ============================================================

def build_simple_table(
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    metrics: list[str],
    precision: int,
    use_ci: bool,
    error_mode: str = "ci95",
) -> str:
    groups = sorted(summary_df[group_col].unique(), key=lambda v: sort_key(group_col, v))

    headers = [group_col] + [metric_label(m) for m in metrics]
    rows: list[list[str]] = []

    for group in groups:
        sub = summary_df[summary_df[group_col] == group]
        row = [group_label(group_col, group)]

        for metric in metrics:
            as_percent = metric in RATE_METRICS
            mean, err = mean_and_error(sub[metric], error_mode=error_mode)
            row.append(
                format_cell(
                    mean,
                    err if use_ci else None,
                    precision=precision,
                    as_percent=as_percent,
                )
            )

        rows.append(row)

    return markdown_table(headers, rows)


def build_pivot_table(
    summary_df: pd.DataFrame,
    *,
    rows_col: str,
    cols_col: str,
    metric: str,
    precision: int,
    use_ci: bool,
    error_mode: str = "ci95",
) -> str:
    row_values = sorted(summary_df[rows_col].unique(), key=lambda v: sort_key(rows_col, v))
    col_values = sorted(summary_df[cols_col].unique(), key=lambda v: sort_key(cols_col, v))

    headers = [rows_col] + [group_label(cols_col, c) for c in col_values]
    rows: list[list[str]] = []

    as_percent = metric in RATE_METRICS

    for r in row_values:
        row = [group_label(rows_col, r)]

        for c in col_values:
            sub = summary_df[(summary_df[rows_col] == r) & (summary_df[cols_col] == c)]

            if len(sub) == 0:
                row.append("--")
                continue

            mean, err = mean_and_error(sub[metric], error_mode=error_mode)
            row.append(
                format_cell(
                    mean,
                    err if use_ci else None,
                    precision=precision,
                    as_percent=as_percent,
                )
            )

        rows.append(row)

    return markdown_table(headers, rows)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a results directory's summary.csv into a Markdown table for rebuttals."
    )

    parser.add_argument("--results-dir", type=str, required=True)

    parser.add_argument(
        "--group-col",
        type=str,
        default="algorithm",
        help="Simple mode: column to group rows by (e.g. algorithm, feedback_m, delta_min).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["final_cumulative_regret", "correct", "completed"],
        help="Simple mode: metrics to report as columns.",
    )

    parser.add_argument(
        "--pivot-metric",
        type=str,
        default=None,
        help="If set, switch to pivot mode: build a --rows x --cols matrix of this single metric.",
    )
    parser.add_argument("--rows", type=str, default=None, help="Pivot mode: row grouping column.")
    parser.add_argument("--cols", type=str, default="algorithm", help="Pivot mode: column grouping column.")

    parser.add_argument("--precision", type=int, default=2, help="Decimal places.")
    parser.add_argument(
        "--error",
        type=str,
        choices=["ci95", "sem"],
        default="ci95",
        help="Error shown after '+/-': 'ci95' (default, +/- 1.96*SEM 95%% CI) "
        "or 'sem' (+/- 1 standard error of the mean).",
    )
    parser.add_argument("--no-ci", action="store_true", help="Report mean only, without any error band.")
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Row values to drop from --exclude-col before building the table "
        "(e.g. --exclude uniform to hide a baseline).",
    )
    parser.add_argument(
        "--exclude-col",
        type=str,
        default="algorithm",
        help="Column that --exclude values are matched against. Default: algorithm.",
    )
    parser.add_argument("--caption", type=str, default=None, help="Optional bold caption line above the table.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the .md file. Default: <results-dir>/rebuttal_table[_<metric>].md",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    summary_df = load_summary(results_dir)

    if args.exclude:
        if args.exclude_col not in summary_df.columns:
            raise ValueError(f"--exclude-col '{args.exclude_col}' not found in summary.csv.")
        summary_df = summary_df[~summary_df[args.exclude_col].isin(args.exclude)]

    use_ci = not args.no_ci

    if args.pivot_metric is not None:
        if args.rows is None:
            raise ValueError("--rows is required when using --pivot-metric.")

        table = build_pivot_table(
            summary_df,
            rows_col=args.rows,
            cols_col=args.cols,
            metric=args.pivot_metric,
            precision=args.precision,
            use_ci=use_ci,
            error_mode=args.error,
        )
        default_name = f"rebuttal_table_{args.pivot_metric}.md"
    else:
        table = build_simple_table(
            summary_df,
            group_col=args.group_col,
            metrics=args.metrics,
            precision=args.precision,
            use_ci=use_ci,
            error_mode=args.error,
        )
        default_name = "rebuttal_table.md"

    text = table
    if args.caption:
        text = f"**{args.caption}**\n\n" + text

    output_path = Path(args.output) if args.output else results_dir / default_name
    output_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    print()
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
