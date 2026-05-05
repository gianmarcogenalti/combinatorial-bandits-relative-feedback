# mnl_experiments.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms import MNLTopKUIF, SahaGaillardMNLTopK, UniformRandomTopK


@dataclass(frozen=True)
class MNLTopKFeedback:
    winner: int
    no_choice: int
    censored_ranks: dict[int, int]


@dataclass(frozen=True)
class MNLRoundResult:
    pulled_arms: tuple[int, ...]
    feedback: MNLTopKFeedback
    expected_reward: float
    instantaneous_regret: float
    cumulative_regret: float
    t: int


@dataclass(frozen=True)
class MNLExperimentConfig:
    name: str = "mnl_default"
    task: str = "mnl_topk"

    n: int = 30
    k: int = 5
    T: int = 100000
    num_seeds: int = 20
    delta: float = 0.05

    algorithms: tuple[str, ...] = ("mnl_topk_uif", "sg_adaptive", "sg_no_choice")

    theta0: Optional[float] = None
    theta0_values: Optional[tuple[float, ...]] = None

    instance: dict[str, Any] = field(default_factory=dict)

    output_dir: Optional[str] = None
    save_every: int = 100

    topk_uif_beta_scale: float = 1.0
    sg_x_scale: float = 2.0


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)

        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            return yaml.safe_load(f)

    raise ValueError("Config file must be .yaml, .yml, or .json.")


def get_named_experiment(raw: dict[str, Any], name: str) -> dict[str, Any]:
    experiments = raw.get("experiments")
    if experiments is None:
        raise ValueError("Config file must contain an `experiments` field.")

    if isinstance(experiments, dict):
        if name not in experiments:
            raise KeyError(f"Unknown experiment {name}. Available: {sorted(experiments)}")
        cfg = dict(experiments[name])
        cfg.setdefault("name", name)
        return cfg

    raise ValueError("`experiments` must be a dict.")


def config_from_file(path: Path, name: str) -> MNLExperimentConfig:
    raw = load_yaml_or_json(path)
    cfg = get_named_experiment(raw, name)

    allowed = set(MNLExperimentConfig.__dataclass_fields__.keys())
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"Unknown MNL config fields: {unknown}")

    if "algorithms" in cfg:
        cfg["algorithms"] = tuple(cfg["algorithms"])

    if "theta0_values" in cfg and cfg["theta0_values"] is not None:
        cfg["theta0_values"] = tuple(float(x) for x in cfg["theta0_values"])

    return MNLExperimentConfig(**cfg)


def normalize_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional.")
    if np.any(theta <= 0):
        raise ValueError("theta entries must be positive.")
    return theta / np.max(theta)


def build_theta(config: MNLExperimentConfig) -> np.ndarray:
    spec = dict(config.instance)
    kind = spec.pop("kind")

    if kind == "explicit":
        theta = np.asarray(spec["theta"], dtype=float)
        if len(theta) != config.n:
            raise ValueError(f"len(theta)={len(theta)} but n={config.n}.")
        return normalize_theta(theta)

    if kind == "geometric":
        ratio = float(spec.get("ratio", 0.9))
        if not (0 < ratio < 1):
            raise ValueError("geometric ratio must be in (0,1).")
        theta = ratio ** np.arange(config.n)
        return normalize_theta(theta)

    if kind == "arith":
        start = float(spec.get("start", 1.0))
        step = float(spec.get("step", 0.02))
        min_value = float(spec.get("min_value", 0.05))
        theta = np.maximum(min_value, start - step * np.arange(config.n))
        return normalize_theta(theta)

    raise ValueError(f"Unknown MNL instance kind: {kind}")


def theta0_values_from_config(config: MNLExperimentConfig) -> list[float]:
    if config.theta0_values is not None:
        return list(config.theta0_values)
    if config.theta0 is not None:
        return [float(config.theta0)]
    return [1.0]


def theta_diagnostics(theta: np.ndarray, *, k: int) -> dict[str, float]:
    sorted_theta = np.sort(theta)[::-1]
    return {
        "theta_k": float(sorted_theta[k - 1]),
        "theta_k_plus_1": float(sorted_theta[k]),
        "boundary_gap": float(sorted_theta[k - 1] - sorted_theta[k]),
        "top_k_mean": float(np.mean(sorted_theta[:k])),
        "suboptimal_mean": float(np.mean(sorted_theta[k:])),
    }


class MNLTopKEnv:
    """
    MNL bandit with top-k objective and no-choice.

    Real arms are 0, ..., n - 1.
    No-choice is represented as arm n.
    """

    def __init__(
        self,
        *,
        theta: np.ndarray,
        theta0: float,
        k: int,
        seed: Optional[int] = None,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float).copy()

        if self.theta.ndim != 1:
            raise ValueError("theta must be one-dimensional.")
        if np.any(self.theta <= 0):
            raise ValueError("theta must be positive.")
        if theta0 <= 0:
            raise ValueError("theta0 must be positive.")

        self.n = len(self.theta)
        self.no_choice = self.n
        self.theta0 = float(theta0)
        self.k = int(k)

        if not (1 <= self.k <= self.n):
            raise ValueError("Require 1 <= k <= n.")

        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.cumulative_regret = 0.0

        self.top_k_arms = tuple(np.argsort(-self.theta, kind="stable")[: self.k])
        self.optimal_reward = self.reward(self.top_k_arms)

    def reward(self, arms) -> float:
        arms = tuple(int(i) for i in arms)
        return float(np.mean(self.theta[list(arms)]))

    def step(self, pulled_arms) -> MNLRoundResult:
        selected = tuple(int(i) for i in pulled_arms)

        if len(selected) != self.k or len(set(selected)) != self.k:
            raise ValueError(f"Expected exactly k={self.k} distinct arms.")
        if any(i < 0 or i >= self.n for i in selected):
            raise ValueError("Pulled arms must be real arms in 0,...,n-1.")

        weights = np.array([self.theta[i] for i in selected] + [self.theta0])
        probs = weights / weights.sum()

        winner_pos = int(self.rng.choice(len(weights), p=probs))

        if winner_pos == len(selected):
            winner = self.no_choice
        else:
            winner = selected[winner_pos]

        ranks = {i: 2 for i in selected}
        ranks[self.no_choice] = 2
        ranks[winner] = 1

        expected_reward = self.reward(selected)
        instantaneous_regret = self.optimal_reward - expected_reward

        self.t += 1
        self.cumulative_regret += instantaneous_regret

        return MNLRoundResult(
            pulled_arms=selected,
            feedback=MNLTopKFeedback(
                winner=int(winner),
                no_choice=self.no_choice,
                censored_ranks=ranks,
            ),
            expected_reward=expected_reward,
            instantaneous_regret=float(instantaneous_regret),
            cumulative_regret=float(self.cumulative_regret),
            t=self.t,
        )


def build_algorithm(
    *,
    name: str,
    n: int,
    k: int,
    T: int,
    delta: float,
    seed: int,
    config: MNLExperimentConfig,
):
    if name == "mnl_topk_uif":
        return MNLTopKUIF(
            n=n,
            k=k,
            T=T,
            delta=delta,
            seed=seed,
            beta_scale=config.topk_uif_beta_scale,
        )

    if name == "sg_adaptive":
        return SahaGaillardMNLTopK(
            n=n,
            k=k,
            T=T,
            adaptive=True,
            seed=seed,
            x_scale=config.sg_x_scale,
        )

    if name == "sg_no_choice":
        return SahaGaillardMNLTopK(
            n=n,
            k=k,
            T=T,
            adaptive=False,
            seed=seed,
            x_scale=config.sg_x_scale,
        )

    if name == "uniform":
        return UniformRandomTopK(n=n, k=k, seed=seed)

    raise ValueError(f"Unknown MNL algorithm: {name}")


def run_single_seed(
    *,
    algorithm_name: str,
    theta: np.ndarray,
    theta0: float,
    config: MNLExperimentConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = MNLTopKEnv(
        theta=theta,
        theta0=theta0,
        k=config.k,
        seed=10_000 + seed,
    )

    algo = build_algorithm(
        name=algorithm_name,
        n=env.n,
        k=config.k,
        T=config.T,
        delta=config.delta,
        seed=20_000 + seed,
        config=config,
    )

    rows = []
    true_top_k = set(env.top_k_arms)

    for _ in range(config.T):
        action = algo.select_action()
        result = env.step(action)
        algo.update(result)

        state = algo.state()
        action_set = set(action)

        if result.t % config.save_every == 0 or result.t == config.T:
            rows.append(
                {
                    "experiment": config.name,
                    "theta0": theta0,
                    "algorithm": algorithm_name,
                    "seed": seed,
                    "round": result.t,
                    "instantaneous_regret": result.instantaneous_regret,
                    "cumulative_regret": result.cumulative_regret,
                    "expected_reward": result.expected_reward,
                    "winner": result.feedback.winner,
                    "no_choice_selected": result.feedback.winner == env.no_choice,
                    "accepted_size": len(state.accepted),
                    "unknown_size": len(state.unknown),
                    "completed": state.completed,
                    "pivot": state.pivot,
                    "action_correct": action_set == true_top_k,
                    "action_precision": len(action_set & true_top_k) / len(action_set),
                    "action_recall": len(action_set & true_top_k) / len(true_top_k),
                }
            )

    history_df = pd.DataFrame(rows)

    final_state = algo.state()
    recommended = set(final_state.accepted)
    last_action = set(action)

    tail_start = int(0.9 * config.T)
    tail = history_df[history_df["round"] >= tail_start]

    extra = {}
    if hasattr(algo, "diagnostics"):
        extra = algo.diagnostics()

    summary = {
        "experiment": config.name,
        "theta0": theta0,
        "algorithm": algorithm_name,
        "seed": seed,
        "n": config.n,
        "k": config.k,
        "T": config.T,
        "final_cumulative_regret": env.cumulative_regret,
        "completed": final_state.completed,
        "recommended_correct": recommended == true_top_k,
        "recommended_precision": (
            len(recommended & true_top_k) / len(recommended) if recommended else 0.0
        ),
        "recommended_recall": len(recommended & true_top_k) / len(true_top_k),
        "last_action_correct": last_action == true_top_k,
        "last_action_precision": len(last_action & true_top_k) / len(last_action),
        "last_action_recall": len(last_action & true_top_k) / len(true_top_k),
        "tail_mean_instantaneous_regret": (
            float(tail["instantaneous_regret"].mean()) if len(tail) else np.nan
        ),
        "tail_action_correct_rate": (
            float(tail["action_correct"].mean()) if len(tail) else np.nan
        ),
        "true_top_k": tuple(sorted(true_top_k)),
        "recommended": tuple(sorted(recommended)),
        "last_action": tuple(sorted(last_action)),
        "num_invariant_violations": extra.get("num_invariant_violations", 0),
    }

    return history_df, summary


def run_experiment(config: MNLExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    theta = build_theta(config)
    theta0_values = theta0_values_from_config(config)

    print("MNL experiment:", config.name)
    print("theta0 values:", theta0_values)
    print("theta diagnostics:", theta_diagnostics(theta, k=config.k))
    print("theta:", np.round(theta, 4))
    print()

    histories = []
    summaries = []

    for theta0 in theta0_values:
        for algorithm_name in config.algorithms:
            for seed in range(config.num_seeds):
                hist, summ = run_single_seed(
                    algorithm_name=algorithm_name,
                    theta=theta,
                    theta0=float(theta0),
                    config=config,
                    seed=seed,
                )

                histories.append(hist)
                summaries.append(summ)

                print(
                    f"theta0={theta0:<8g} {algorithm_name:14s} seed={seed:03d} "
                    f"regret={summ['final_cumulative_regret']:.4f} "
                    f"last_correct={summ['last_action_correct']} "
                    f"tail_regret={summ['tail_mean_instantaneous_regret']:.5f}"
                )

    return pd.concat(histories, ignore_index=True), pd.DataFrame(summaries), theta


def aggregate_curve(df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["theta0", "algorithm", "round"])[value_col]
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
    value_col: str,
    output_col: str,
    window: int,
) -> pd.DataFrame:
    df = df.sort_values(["theta0", "algorithm", "seed", "round"]).copy()

    df[output_col] = (
        df.groupby(["theta0", "algorithm", "seed"])[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def downsample_rounds(df: pd.DataFrame, *, stride: Optional[int]) -> pd.DataFrame:
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


def plot_curves_by_theta0(
    history_df: pd.DataFrame,
    *,
    output_dir: Path,
    value_col: str,
    y_label: str,
    file_prefix: str,
    stride: Optional[int],
) -> None:
    agg = aggregate_curve(history_df, value_col=value_col)
    agg = downsample_rounds(agg, stride=stride)

    for theta0, sub_theta in agg.groupby("theta0"):
        plt.figure(figsize=(8, 5))

        for algorithm_name, sub in sub_theta.groupby("algorithm"):
            x = sub["round"].to_numpy()
            y = sub["mean"].to_numpy()
            ci = sub["ci95"].to_numpy()

            plt.plot(x, y, label=algorithm_name)
            plt.fill_between(x, y - ci, y + ci, alpha=0.2)

        plt.xlabel("Round")
        plt.ylabel(y_label)
        plt.title(f"{y_label}, theta0={theta0:g}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{file_prefix}_theta0_{theta0:g}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_instantaneous_by_theta0(
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

    plot_curves_by_theta0(
        df,
        output_dir=output_dir,
        value_col="instantaneous_regret_ma",
        y_label=f"Instantaneous top-k regret, MA window={window}",
        file_prefix="instantaneous_regret",
        stride=stride,
    )


def plot_final_regret_vs_theta0(summary_df: pd.DataFrame, *, output_dir: Path) -> None:
    grouped = (
        summary_df.groupby(["theta0", "algorithm"])["final_cumulative_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    plt.figure(figsize=(8, 5))

    for algorithm_name, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("theta0")
        x = sub["theta0"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        plt.errorbar(x, y, yerr=ci, marker="o", capsize=4, label=algorithm_name)

    plt.xscale("log")
    plt.xlabel("No-choice parameter theta0")
    plt.ylabel("Final cumulative top-k regret")
    plt.title("Final regret vs theta0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_regret_vs_theta0.pdf", bbox_inches="tight")
    plt.close()


def plot_tail_regret_vs_theta0(summary_df: pd.DataFrame, *, output_dir: Path) -> None:
    grouped = (
        summary_df.groupby(["theta0", "algorithm"])["tail_mean_instantaneous_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["stderr"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["stderr"]

    plt.figure(figsize=(8, 5))

    for algorithm_name, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("theta0")
        x = sub["theta0"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        plt.errorbar(x, y, yerr=ci, marker="o", capsize=4, label=algorithm_name)

    plt.xscale("log")
    plt.xlabel("No-choice parameter theta0")
    plt.ylabel("Tail mean instantaneous regret")
    plt.title("Tail instantaneous regret vs theta0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tail_instantaneous_regret_vs_theta0.pdf", bbox_inches="tight")
    plt.close()


def plot_last_action_success(summary_df: pd.DataFrame, *, output_dir: Path) -> None:
    grouped = (
        summary_df.groupby(["theta0", "algorithm"])["last_action_correct"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for algorithm_name, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("theta0")

        plt.plot(
            sub["theta0"],
            sub["last_action_correct"],
            marker="o",
            label=algorithm_name,
        )

    plt.xscale("log")
    plt.xlabel("No-choice parameter theta0")
    plt.ylabel("Last-action exact top-k rate")
    plt.ylim(0, 1.05)
    plt.title("Last-action success vs theta0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "last_action_success_vs_theta0.pdf", bbox_inches="tight")
    plt.close()


def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df.groupby(["theta0", "algorithm"])
        .agg(
            mean_final_regret=("final_cumulative_regret", "mean"),
            std_final_regret=("final_cumulative_regret", "std"),
            last_action_success=("last_action_correct", "mean"),
            mean_tail_instantaneous_regret=("tail_mean_instantaneous_regret", "mean"),
            recommended_success=("recommended_correct", "mean"),
            completion_rate=("completed", "mean"),
        )
        .reset_index()
    )


def save_outputs(
    *,
    config: MNLExperimentConfig,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    theta: np.ndarray,
    stride: Optional[int],
) -> None:
    output_dir = Path(config.output_dir or f"results/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_dir / "history.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    agg = aggregate_summary(summary_df)
    agg.to_csv(output_dir / "aggregate_summary.csv", index=False)

    np.savetxt(output_dir / "theta.csv", theta, delimiter=",")

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(config),
                "theta": theta.tolist(),
                "theta_diagnostics": theta_diagnostics(theta, k=config.k),
            },
            f,
            indent=2,
        )

    pdf_dir = output_dir / "pdf_plots"
    pdf_dir.mkdir(exist_ok=True)

    num_saved_points = history_df["round"].nunique()
    window = max(1, min(500, num_saved_points // 30))

    plot_curves_by_theta0(
        history_df,
        output_dir=pdf_dir,
        value_col="cumulative_regret",
        y_label="Cumulative top-k regret",
        file_prefix="cumulative_regret",
        stride=stride,
    )

    plot_instantaneous_by_theta0(
        history_df,
        output_dir=pdf_dir,
        window=window,
        stride=stride,
    )

    plot_final_regret_vs_theta0(summary_df, output_dir=pdf_dir)
    plot_tail_regret_vs_theta0(summary_df, output_dir=pdf_dir)
    plot_last_action_success(summary_df, output_dir=pdf_dir)

    print()
    print("Aggregate summary:")
    print(agg.to_string(index=False))
    print()
    print(f"Saved outputs to: {output_dir}")
    print(f"Saved PDF plots to: {pdf_dir}")


def validate_config(config: MNLExperimentConfig) -> None:
    if config.task != "mnl_topk":
        raise ValueError("This script only supports task: mnl_topk.")

    if not (1 <= config.k <= config.n):
        raise ValueError("Require 1 <= k <= n.")

    if config.T < 1:
        raise ValueError("T must be positive.")

    if config.num_seeds < 1:
        raise ValueError("num_seeds must be positive.")

    if config.save_every < 1:
        raise ValueError("save_every must be positive.")

    if not config.instance:
        raise ValueError("Missing instance config.")

    allowed = {"mnl_topk_uif", "sg_adaptive", "sg_no_choice", "uniform"}
    unknown = set(config.algorithms) - allowed

    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="experiment_configs.yaml")
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Downsample plotted curves: keep one plotted point every stride rounds.",
    )

    args = parser.parse_args()

    config = config_from_file(Path(args.config), args.name)
    validate_config(config)

    history_df, summary_df, theta = run_experiment(config)

    save_outputs(
        config=config,
        history_df=history_df,
        summary_df=summary_df,
        theta=theta,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()