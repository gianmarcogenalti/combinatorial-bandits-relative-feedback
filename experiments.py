# experiments.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import RelativeFeedbackCMABEnv
from algorithms import TopKUIF, UniformRandomTopK, RecMaxMinUCB


# ============================================================
# Config dataclass
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "default"

    n: int = 20
    k: int = 5
    m: int = 5
    T: int = 5_000
    num_seeds: int = 20
    delta: float = 0.05

    algorithms: tuple[str, ...] = ("topk_uif", "rec_maxmin_ucb", "uniform")

    # Instance is now specified as a dictionary from YAML/JSON.
    # Examples:
    #   {"kind": "geometric", "ratio": 0.9}
    #   {"kind": "weak_kth", "weak_optimal_value": 0.52, ...}
    #   {"kind": "explicit", "theta": [1.0, 0.9, ...]}
    instance: dict[str, Any] = field(
        default_factory=lambda: {"kind": "geometric", "ratio": 0.9}
    )

    output_dir: Optional[str] = None
    save_every: int = 1

    recmaxmin_alpha: float = 0.51
    allow_recmaxmin_topm_variant: bool = False


# ============================================================
# Config loading
# ============================================================

def load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Reading YAML configs requires PyYAML. Install with:\n\n"
                    "    pip install pyyaml\n"
                ) from exc

            return yaml.safe_load(f)

    raise ValueError("Config file must be .json, .yaml, or .yml")


def list_experiment_names(config_path: Path) -> list[str]:
    raw = load_yaml_or_json(config_path)

    experiments = raw.get("experiments")
    if experiments is None:
        raise ValueError("Config file must contain an 'experiments' field.")

    if isinstance(experiments, dict):
        return sorted(experiments.keys())

    if isinstance(experiments, list):
        return sorted(exp["name"] for exp in experiments)

    raise ValueError("'experiments' must be either a mapping or a list.")


def get_named_experiment(raw: dict[str, Any], name: str) -> dict[str, Any]:
    experiments = raw.get("experiments")
    if experiments is None:
        raise ValueError("Config file must contain an 'experiments' field.")

    if isinstance(experiments, dict):
        if name not in experiments:
            available = ", ".join(sorted(experiments.keys()))
            raise KeyError(f"Unknown experiment '{name}'. Available: {available}")

        cfg = dict(experiments[name])
        cfg.setdefault("name", name)
        return cfg

    if isinstance(experiments, list):
        for exp in experiments:
            if exp.get("name") == name:
                return dict(exp)

        available = ", ".join(sorted(exp["name"] for exp in experiments))
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}")

    raise ValueError("'experiments' must be either a mapping or a list.")


def config_from_file(config_path: Path, name: str) -> ExperimentConfig:
    raw = load_yaml_or_json(config_path)
    cfg_dict = get_named_experiment(raw, name)

    allowed = set(ExperimentConfig.__dataclass_fields__.keys())
    unknown = sorted(set(cfg_dict.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown config fields in '{name}': {unknown}")

    if "algorithms" in cfg_dict:
        cfg_dict["algorithms"] = tuple(cfg_dict["algorithms"])

    return ExperimentConfig(**cfg_dict)


# ============================================================
# Instance generation
# ============================================================

def normalize_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional.")
    if np.any(theta <= 0):
        raise ValueError("All theta values must be positive.")
    return theta / np.max(theta)


def maybe_shuffle(theta: np.ndarray, *, shuffle: bool, seed: Optional[int]) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).copy()

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(theta)

    return theta


def make_geometric_theta(
    *,
    n: int,
    ratio: float,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    if not (0 < ratio < 1):
        raise ValueError("geometric ratio must be in (0, 1).")

    theta = ratio ** np.arange(n)
    theta = normalize_theta(theta)

    return maybe_shuffle(theta, shuffle=shuffle, seed=seed)


def make_gap_theta(
    *,
    n: int,
    k: int,
    top_value: float = 1.0,
    boundary_gap: float = 0.1,
    tail_value: Optional[float] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    if not (1 <= k < n):
        raise ValueError("Require 1 <= k < n.")
    if top_value <= 0:
        raise ValueError("top_value must be positive.")
    if not (0 < boundary_gap < top_value):
        raise ValueError("boundary_gap must be in (0, top_value).")

    if tail_value is None:
        tail_value = top_value - boundary_gap

    if not (0 < tail_value < top_value):
        raise ValueError("tail_value must be in (0, top_value).")

    theta = np.empty(n, dtype=float)
    theta[:k] = top_value
    theta[k:] = tail_value
    theta = normalize_theta(theta)

    return maybe_shuffle(theta, shuffle=shuffle, seed=seed)


def make_weak_kth_theta(
    *,
    n: int,
    k: int,
    strong_value: float = 1.0,
    weak_optimal_value: float = 0.52,
    decoy_value: float = 0.50,
    tail_value: float = 0.15,
    num_decoys: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Pathological family:

        k-1 strong optimal arms,
        1 weak optimal arm,
        many near-optimal decoys,
        remaining bad tail arms.
    """

    if not (1 < k < n):
        raise ValueError("Require 1 < k < n.")
    if not (strong_value > weak_optimal_value > decoy_value > tail_value > 0):
        raise ValueError(
            "Require strong_value > weak_optimal_value > decoy_value > tail_value > 0."
        )

    max_decoys = n - k
    if num_decoys is None:
        num_decoys = max_decoys

    if not (1 <= num_decoys <= max_decoys):
        raise ValueError(f"num_decoys must be in [1, {max_decoys}].")

    num_tail = n - (k - 1) - 1 - num_decoys

    theta = np.array(
        [strong_value] * (k - 1)
        + [weak_optimal_value]
        + [decoy_value] * num_decoys
        + [tail_value] * num_tail,
        dtype=float,
    )

    theta = normalize_theta(theta)

    return maybe_shuffle(theta, shuffle=shuffle, seed=seed)


def make_boundary_cloud_theta(
    *,
    n: int,
    k: int,
    top_start: float = 1.0,
    boundary_value: float = 0.50,
    gap: float = 0.02,
    cloud_size: Optional[int] = None,
    tail_value: float = 0.15,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Smooth instance with many arms near the top-k boundary.
    """

    if not (1 < k < n):
        raise ValueError("Require 1 < k < n.")
    if not (top_start > boundary_value + gap > boundary_value > tail_value > 0):
        raise ValueError(
            "Require top_start > boundary_value + gap > boundary_value > tail_value > 0."
        )

    max_cloud = n - k
    if cloud_size is None:
        cloud_size = max_cloud

    if not (1 <= cloud_size <= max_cloud):
        raise ValueError(f"cloud_size must be in [1, {max_cloud}].")

    top_values = np.linspace(top_start, boundary_value + gap, k)
    cloud_values = np.full(cloud_size, boundary_value)
    tail_values = np.full(n - k - cloud_size, tail_value)

    theta = np.concatenate([top_values, cloud_values, tail_values])
    theta = normalize_theta(theta)

    return maybe_shuffle(theta, shuffle=shuffle, seed=seed)


def make_two_level_decoy_theta(
    *,
    n: int,
    k: int,
    strong_value: float = 1.0,
    weak_optimal_value: float = 0.60,
    close_decoy_value: float = 0.58,
    bad_decoy_value: float = 0.20,
    close_decoy_fraction: float = 0.5,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Some decoys are close to the boundary; other decoys are clearly bad.
    """

    if not (1 < k < n):
        raise ValueError("Require 1 < k < n.")
    if not (
        strong_value > weak_optimal_value > close_decoy_value > bad_decoy_value > 0
    ):
        raise ValueError(
            "Require strong > weak_optimal > close_decoy > bad_decoy > 0."
        )
    if not (0 < close_decoy_fraction <= 1):
        raise ValueError("close_decoy_fraction must be in (0, 1].")

    num_suboptimal = n - k
    num_close = max(1, int(round(close_decoy_fraction * num_suboptimal)))
    num_bad = num_suboptimal - num_close

    theta = np.array(
        [strong_value] * (k - 1)
        + [weak_optimal_value]
        + [close_decoy_value] * num_close
        + [bad_decoy_value] * num_bad,
        dtype=float,
    )

    theta = normalize_theta(theta)

    return maybe_shuffle(theta, shuffle=shuffle, seed=seed)


def build_theta(config: ExperimentConfig) -> np.ndarray:
    spec = dict(config.instance)
    kind = spec.pop("kind")

    if kind == "explicit":
        if "theta" not in spec:
            raise ValueError("Explicit instance requires field 'theta'.")

        theta = normalize_theta(np.asarray(spec["theta"], dtype=float))

        if len(theta) != config.n:
            raise ValueError(
                f"Explicit theta has length {len(theta)}, but config n={config.n}."
            )

        return theta

    if kind == "geometric":
        return make_geometric_theta(n=config.n, **spec)

    if kind == "gap":
        return make_gap_theta(n=config.n, k=config.k, **spec)

    if kind == "weak_kth":
        return make_weak_kth_theta(n=config.n, k=config.k, **spec)

    if kind == "boundary_cloud":
        return make_boundary_cloud_theta(n=config.n, k=config.k, **spec)

    if kind == "two_level_decoy":
        return make_two_level_decoy_theta(n=config.n, k=config.k, **spec)

    raise ValueError(
        f"Unknown instance kind '{kind}'. "
        "Supported kinds: explicit, geometric, gap, weak_kth, "
        "boundary_cloud, two_level_decoy."
    )


def theta_diagnostics(theta: np.ndarray, *, k: int) -> dict[str, float]:
    sorted_theta = np.sort(theta)[::-1]

    return {
        "theta_k": float(sorted_theta[k - 1]),
        "theta_k_plus_1": float(sorted_theta[k]),
        "boundary_gap": float(sorted_theta[k - 1] - sorted_theta[k]),
        "top_k_mean": float(np.mean(sorted_theta[:k])),
        "suboptimal_mean": float(np.mean(sorted_theta[k:])),
        "theta_min": float(np.min(theta)),
        "theta_max": float(np.max(theta)),
    }


# ============================================================
# Algorithms
# ============================================================

def build_algorithm(
    *,
    name: str,
    n: int,
    k: int,
    m: int,
    T: int,
    delta: float,
    seed: int,
    config: ExperimentConfig,
):
    if name == "topk_uif":
        return TopKUIF(
            n=n,
            k=k,
            m=m,
            T=T,
            delta=delta,
            seed=seed,
        )

    if name == "rec_maxmin_ucb":
        require_full_ranking = not config.allow_recmaxmin_topm_variant

        if require_full_ranking and m != k:
            raise ValueError(
                "Faithful RecMaxMinUCB requires full ranking feedback, m == k. "
                f"Current values: k={k}, m={m}."
            )

        return RecMaxMinUCB(
            n=n,
            k=k,
            m=m,
            alpha=config.recmaxmin_alpha,
            seed=seed,
            require_full_ranking=require_full_ranking,
        )

    if name == "uniform":
        return UniformRandomTopK(
            n=n,
            k=k,
            seed=seed,
        )

    raise ValueError(f"Unknown algorithm: {name}")


# ============================================================
# Running experiments
# ============================================================

def run_single_seed(
    *,
    algorithm_name: str,
    theta: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = RelativeFeedbackCMABEnv(
        theta=theta,
        k=config.k,
        m=config.m,
        seed=10_000 + seed,
        normalize_theta=True,
        require_unique_top_k=False,
    )

    algo = build_algorithm(
        name=algorithm_name,
        n=env.n,
        k=config.k,
        m=config.m,
        T=config.T,
        delta=config.delta,
        seed=20_000 + seed,
        config=config,
    )

    rows: list[dict[str, Any]] = []

    for _ in range(config.T):
        action = algo.select_action()
        result = env.step(action)
        algo.update(result)

        state = algo.state()

        if result.t % config.save_every == 0 or result.t == config.T:
            rows.append(
                {
                    "experiment": config.name,
                    "algorithm": algorithm_name,
                    "seed": seed,
                    "round": result.t,
                    "instantaneous_regret": result.instantaneous_regret,
                    "cumulative_regret": result.cumulative_regret,
                    "expected_reward": result.expected_reward,
                    "accepted_size": len(state.accepted),
                    "unknown_size": len(state.unknown),
                    "completed": state.completed,
                    "pivot": state.pivot,
                }
            )

    final_state = algo.state()

    true_top_k = set(env.top_k_arms)
    recommended = set(final_state.accepted)

    precision = (
        len(recommended & true_top_k) / len(recommended)
        if len(recommended) > 0
        else 0.0
    )
    recall = len(recommended & true_top_k) / len(true_top_k)

    summary = {
        "experiment": config.name,
        "algorithm": algorithm_name,
        "seed": seed,
        "n": config.n,
        "k": config.k,
        "m": config.m,
        "T": config.T,
        "final_cumulative_regret": env.cumulative_regret,
        "completed": final_state.completed,
        "correct": recommended == true_top_k,
        "recommended_size": len(recommended),
        "precision": precision,
        "recall": recall,
        "true_top_k": tuple(sorted(true_top_k)),
        "recommended": tuple(sorted(recommended)),
    }

    return pd.DataFrame(rows), summary


def run_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    theta = build_theta(config)

    print("Experiment:", config.name)
    print("Theta diagnostics:", theta_diagnostics(theta, k=config.k))
    print("Theta:", np.round(theta, 4))
    print()

    histories: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for algorithm_name in config.algorithms:
        for seed in range(config.num_seeds):
            history_df, summary = run_single_seed(
                algorithm_name=algorithm_name,
                theta=theta,
                config=config,
                seed=seed,
            )

            histories.append(history_df)
            summaries.append(summary)

            print(
                f"{algorithm_name:16s} seed={seed:03d} "
                f"final_regret={summary['final_cumulative_regret']:.4f} "
                f"correct={summary['correct']} "
                f"completed={summary['completed']} "
                f"recall={summary['recall']:.3f}"
            )

    return (
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(summaries),
        theta,
    )


# ============================================================
# Aggregation and plotting
# ============================================================

def aggregate_curve(history_df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    grouped = (
        history_df
        .groupby(["algorithm", "round"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = grouped["stderr"]

    return grouped


def summarize(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df
        .groupby("algorithm")
        .agg(
            mean_final_regret=("final_cumulative_regret", "mean"),
            std_final_regret=("final_cumulative_regret", "std"),
            median_final_regret=("final_cumulative_regret", "median"),
            correctness_rate=("correct", "mean"),
            completion_rate=("completed", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .reset_index()
    )
def add_moving_average(
    history_df: pd.DataFrame,
    *,
    value_col: str,
    output_col: str,
    window: int,
) -> pd.DataFrame:
    df = history_df.sort_values(["algorithm", "seed", "round"]).copy()

    df[output_col] = (
        df
        .groupby(["algorithm", "seed"])[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df

def plot_instantaneous_regret(
    history_df: pd.DataFrame,
    *,
    output_path: Path,
    window: int = 200,
) -> None:
    """
    Plot moving-average instantaneous top-k regret with 95% CI.

    The moving average is computed separately for each algorithm/seed,
    then averaged across seeds.
    """

    df = add_moving_average(
        history_df,
        value_col="instantaneous_regret",
        output_col="instantaneous_regret_ma",
        window=window,
    )

    agg = aggregate_curve(df, value_col="instantaneous_regret_ma")

    plt.figure(figsize=(8, 5))

    for algorithm_name, sub in agg.groupby("algorithm"):
        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        plt.plot(x, y, label=algorithm_name)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel(f"Instantaneous top-k regret, moving average window={window}")
    plt.title("Instantaneous top-k regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_cumulative_regret(history_df: pd.DataFrame, *, output_path: Path) -> None:
    agg = aggregate_curve(history_df, value_col="cumulative_regret")

    plt.figure(figsize=(8, 5))

    for algorithm_name, sub in agg.groupby("algorithm"):
        x = sub["round"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        plt.plot(x, y, label=algorithm_name)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel("Cumulative top-k regret")
    plt.title("Cumulative top-k regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_final_regret(summary_df: pd.DataFrame, *, output_path: Path) -> None:
    algorithms = list(summary_df["algorithm"].unique())

    data = [
        summary_df.loc[
            summary_df["algorithm"] == algorithm,
            "final_cumulative_regret",
        ].to_numpy()
        for algorithm in algorithms
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=algorithms)
    plt.ylabel("Final cumulative top-k regret")
    plt.title("Final regret across seeds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_correctness(summary_df: pd.DataFrame, *, output_path: Path) -> None:
    grouped = (
        summary_df
        .groupby("algorithm")
        .agg(
            correctness_rate=("correct", "mean"),
            completion_rate=("completed", "mean"),
        )
        .reset_index()
    )

    algorithms = grouped["algorithm"].to_list()
    x = np.arange(len(algorithms))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, grouped["correctness_rate"], width, label="correct")
    plt.bar(x + width / 2, grouped["completion_rate"], width, label="completed")
    plt.xticks(x, algorithms)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Fraction of seeds")
    plt.title("Identification success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_outputs(
    *,
    config: ExperimentConfig,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    theta: np.ndarray,
) -> None:
    output_dir = Path(config.output_dir or f"results/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_dir / "history.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    aggregate_summary = summarize(summary_df)
    aggregate_summary.to_csv(output_dir / "aggregate_summary.csv", index=False)

    np.savetxt(output_dir / "theta.csv", theta, delimiter=",")

    metadata = {
        "config": asdict(config),
        "theta_diagnostics": theta_diagnostics(theta, k=config.k),
        "theta": theta.tolist(),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plot_cumulative_regret(
        history_df,
        output_path=output_dir / "cumulative_regret.png",
    )

    num_saved_points = history_df["round"].nunique()
    instant_window = max(1, min(200, num_saved_points // 20))

    plot_instantaneous_regret(
        history_df,
        output_path=output_dir / "instantaneous_regret.png",
        window=instant_window,
    )

    plot_final_regret(
        summary_df,
        output_path=output_dir / "final_regret_boxplot.png",
    )

    plot_correctness(
        summary_df,
        output_path=output_dir / "correctness_completion.png",
    )

    print()
    print("Saved outputs to:", output_dir)
    print("  history.csv")
    print("  summary.csv")
    print("  aggregate_summary.csv")
    print("  theta.csv")
    print("  metadata.json")
    print("  cumulative_regret.png")
    print("  final_regret_boxplot.png")
    print("  correctness_completion.png")


# ============================================================
# Validation and CLI
# ============================================================

def validate_config(config: ExperimentConfig) -> None:
    if config.n < 2:
        raise ValueError("n must be at least 2.")

    if not (1 <= config.m <= config.k <= config.n):
        raise ValueError("Require 1 <= m <= k <= n.")

    if config.T < 1:
        raise ValueError("T must be at least 1.")

    if config.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1.")

    if config.save_every < 1:
        raise ValueError("save_every must be at least 1.")

    if not (0 < config.delta <= 1):
        raise ValueError("delta must be in (0, 1].")

    allowed_algorithms = {"topk_uif", "rec_maxmin_ucb", "uniform"}
    unknown_algorithms = set(config.algorithms) - allowed_algorithms

    if unknown_algorithms:
        raise ValueError(f"Unknown algorithms: {unknown_algorithms}")

    if (
        "rec_maxmin_ucb" in config.algorithms
        and config.m != config.k
        and not config.allow_recmaxmin_topm_variant
    ):
        raise ValueError(
            "Faithful RecMaxMinUCB requires m == k. "
            f"Current values: k={config.k}, m={config.m}. "
            "Either set m=k, remove rec_maxmin_ucb, or set "
            "allow_recmaxmin_topm_variant: true."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="experiment_configs.yaml",
        help="Path to YAML/JSON file containing named experiments.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of experiment to run from the config file.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment names and exit.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if args.list:
        names = list_experiment_names(config_path)
        print("Available experiments:")
        for name in names:
            print(" ", name)
        return

    if args.name is None:
        raise ValueError(
            "Please specify an experiment name, for example:\n\n"
            "    python experiments.py --name weak_kth_pathological\n"
        )

    config = config_from_file(config_path, args.name)
    validate_config(config)

    history_df, summary_df, theta = run_experiment(config)

    print()
    print("Aggregate summary:")
    print(summarize(summary_df).to_string(index=False))

    save_outputs(
        config=config,
        history_df=history_df,
        summary_df=summary_df,
        theta=theta,
    )


if __name__ == "__main__":
    main()