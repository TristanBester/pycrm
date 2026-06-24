import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from examples.rm.discrete.core import (
    PuckWorld,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
    PuckWorldRewardMachine,
)
from pycrm.agents.sbx.dqn import DQN as SBXDQN
from pycrm.agents.sbx.dqn import CounterfactualDQN as SBXCounterfactualDQN

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
RESULTS_DIR = SCRIPT_DIR / "results" / "jax_validation"

ALGORITHMS = ("dqn", "cdqn")
BASELINE_LOGS = {
    "dqn": LOG_DIR / "DQN_1",
    "cdqn": LOG_DIR / "C-DQN_2",
}
SBX_LOG_NAMES = {
    "dqn": "SBX-DQN",
    "cdqn": "SBX-C-DQN",
}


def make_env(seed: int, max_steps: int) -> PuckWorldCrossProduct:
    """Create the RM-discrete PuckWorld environment used by the SB3 examples."""
    np.random.seed(seed)
    ground_env = PuckWorld()
    labelling_function = PuckWorldLabellingFunction()
    reward_machine = PuckWorldRewardMachine()
    env = PuckWorldCrossProduct(
        ground_env=ground_env,
        machine=reward_machine,
        lf=labelling_function,
        max_steps=max_steps,
    )
    env.reset(seed=seed)
    return env


def run_sbx_algorithm(
    *,
    algorithm: str,
    seed: int,
    total_timesteps: int,
    max_steps: int,
    learning_rate: float,
    exploration_fraction: float | None,
    exploration_final_eps: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    train_freq: int,
    gradient_steps: int,
    target_update_interval: int,
    policy_net_arch: list[int],
    device: str,
    tensorboard_log: Path,
    tb_log_suffix: str | None,
) -> dict[str, Any]:
    """Run the SBX/JAX DQN variant with the SB3 example hyperparameters."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    env = make_env(seed=seed, max_steps=max_steps)
    env.action_space.seed(seed)

    if exploration_fraction is None:
        exploration_fraction = 0.2 if algorithm == "cdqn" else 0.1
    agent_class = SBXCounterfactualDQN if algorithm == "cdqn" else SBXDQN
    tb_log_name = SBX_LOG_NAMES[algorithm]
    if tb_log_suffix:
        tb_log_name = f"{tb_log_name}-{tb_log_suffix}"
    before = _matching_log_dirs(tensorboard_log, tb_log_name)
    model = agent_class(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=str(tensorboard_log),
        verbose=1,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        buffer_size=buffer_size,
        batch_size=batch_size,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        policy_kwargs={"net_arch": policy_net_arch},
        device=device,
        seed=seed,
    )

    started_at = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        tb_log_name=tb_log_name,
    )
    train_time_s = time.perf_counter() - started_at
    env.close()

    after = _matching_log_dirs(tensorboard_log, tb_log_name)
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    log_dir = (
        new_dirs[-1]
        if new_dirs
        else _latest_log_dir(tensorboard_log, tb_log_name)
    )
    return {
        "algorithm": algorithm,
        "backend": "sbx",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "max_steps": max_steps,
        "train_time_s": float(train_time_s),
        "tensorboard_log": str(log_dir),
        "settings": {
            "learning_rate": learning_rate,
            "learning_starts": learning_starts,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "target_update_interval": target_update_interval,
            "policy_kwargs": {"net_arch": policy_net_arch},
            "device": device,
            "tb_log_name": tb_log_name,
        },
    }


def compare_runs(
    *,
    sbx_runs: list[dict[str, Any]],
    baseline_logs: dict[str, Path],
    results_dir: Path,
) -> dict[str, Any]:
    """Extract SB3/SBX TensorBoard curves and write comparison artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    needed_algorithms = {run["algorithm"] for run in sbx_runs}

    for algorithm in ALGORITHMS:
        if algorithm not in needed_algorithms:
            continue
        baseline_log = baseline_logs[algorithm]
        if not baseline_log.exists():
            raise FileNotFoundError(
                f"Missing SB3 baseline TensorBoard log for {algorithm}: "
                f"{baseline_log}"
            )
        runs.append(
            _load_tensorboard_run(
                algorithm=algorithm,
                backend="sb3",
                log_dir=baseline_log,
            )
        )

    for run in sbx_runs:
        runs.append(
            _load_tensorboard_run(
                algorithm=run["algorithm"],
                backend="sbx",
                log_dir=Path(run["tensorboard_log"]),
                metadata=run,
            )
        )

    result = {
        "summary": _summarize_comparison(runs),
        "runs": runs,
        "note": (
            "SBX/JAX runs use the RM-discrete PuckWorld environment and the same "
            "training horizon and DQN hyperparameters as the SB3 example scripts."
        ),
    }

    json_path = results_dir / "comparison.json"
    plot_path = results_dir / "rollout_ep_rew_mean.png"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot_comparison(runs, plot_path)
    result["artifacts"] = {
        "json": str(json_path),
        "plot": str(plot_path),
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _load_tensorboard_run(
    *,
    algorithm: str,
    backend: str,
    log_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scalars = _load_scalars(log_dir)
    return {
        "algorithm": algorithm,
        "backend": backend,
        "tensorboard_log": str(log_dir),
        "metadata": metadata or {},
        "scalars": scalars,
    }


def _load_scalars(log_dir: Path) -> dict[str, list[dict[str, float | int]]]:
    event_file = _latest_event_file(log_dir)
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [
            {
                "step": int(event.step),
                "wall_time": float(event.wall_time),
                "value": float(event.value),
            }
            for event in accumulator.Scalars(tag)
        ]
    return scalars


def _latest_event_file(log_dir: Path) -> Path:
    event_files = sorted(
        log_dir.glob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {log_dir}")
    return event_files[-1]


def _matching_log_dirs(log_root: Path, tb_log_name: str) -> set[Path]:
    if not log_root.exists():
        return set()
    return {
        path
        for path in log_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{tb_log_name}_")
    }


def _latest_log_dir(log_root: Path, tb_log_name: str) -> Path:
    candidates = sorted(
        _matching_log_dirs(log_root, tb_log_name),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard log dir found for {tb_log_name}")
    return candidates[-1]


def _summarize_comparison(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_algorithm = {
        run["algorithm"]: _curve_stats(run)
        for run in runs
        if run["backend"] == "sb3"
    }
    summary = []
    for run in runs:
        stats = _curve_stats(run)
        row = {
            "algorithm": run["algorithm"],
            "backend": run["backend"],
            "tensorboard_log": run["tensorboard_log"],
            **stats,
        }
        baseline = baseline_by_algorithm.get(run["algorithm"])
        if baseline and run["backend"] != "sb3":
            if row["complete"] is False:
                row["final_delta_vs_sb3"] = None
                row["last20_delta_vs_sb3"] = None
            else:
                row["final_delta_vs_sb3"] = (
                    None
                    if row["final_ep_rew_mean"] is None
                    else float(
                        row["final_ep_rew_mean"] - baseline["final_ep_rew_mean"]
                    )
                )
                row["last20_delta_vs_sb3"] = (
                    None
                    if row["last20_ep_rew_mean"] is None
                    else float(
                        row["last20_ep_rew_mean"] - baseline["last20_ep_rew_mean"]
                    )
                )
        summary.append(row)
    return summary


def _curve_stats(run: dict[str, Any]) -> dict[str, float | int | None]:
    curve = run["scalars"].get("rollout/ep_rew_mean", [])
    metadata = run.get("metadata", {})
    total_timesteps = metadata.get("total_timesteps")
    max_steps = metadata.get("max_steps", 0)
    if not curve:
        return {
            "points": 0,
            "first_step": None,
            "final_step": None,
            "first_ep_rew_mean": None,
            "final_ep_rew_mean": None,
            "last20_ep_rew_mean": None,
            "complete": False if total_timesteps is not None else None,
        }
    values = np.array([point["value"] for point in curve], dtype=np.float64)
    final_step = int(curve[-1]["step"])
    complete = None
    if total_timesteps is not None:
        complete = final_step >= int(total_timesteps) - int(max_steps)
    return {
        "points": len(curve),
        "first_step": int(curve[0]["step"]),
        "final_step": final_step,
        "first_ep_rew_mean": float(values[0]),
        "final_ep_rew_mean": float(values[-1]),
        "last20_ep_rew_mean": float(np.mean(values[-20:])),
        "complete": complete,
    }


def _plot_comparison(runs: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        ("sb3", "dqn"): "#4C78A8",
        ("sbx", "dqn"): "#F58518",
        ("sb3", "cdqn"): "#54A24B",
        ("sbx", "cdqn"): "#B279A2",
    }
    labels = {
        ("sb3", "dqn"): "SB3 DQN",
        ("sbx", "dqn"): "SBX/JAX DQN",
        ("sb3", "cdqn"): "SB3 C-DQN",
        ("sbx", "cdqn"): "SBX/JAX C-DQN",
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for run in runs:
        curve = run["scalars"].get("rollout/ep_rew_mean", [])
        if not curve:
            continue
        key = (run["backend"], run["algorithm"])
        steps = np.array([point["step"] for point in curve], dtype=np.float64)
        values = np.array([point["value"] for point in curve], dtype=np.float64)
        smooth = _moving_average(values, window=10)
        smooth_steps = steps[len(steps) - len(smooth) :]
        ax.plot(steps, values, color=colors[key], alpha=0.18, linewidth=0.8)
        ax.plot(
            smooth_steps,
            smooth,
            color=colors[key],
            label=labels[key],
            linewidth=2.0,
        )

    ax.set_title("RM-Discrete PuckWorld DQN Validation")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("TensorBoard rollout/ep_rew_mean")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _expand_algorithms(algorithms: Iterable[str]) -> list[str]:
    selected = list(algorithms)
    if "all" in selected:
        return list(ALGORITHMS)
    unknown = set(selected) - set(ALGORITHMS)
    if unknown:
        raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
    return selected


def _parse_net_arch(value: str) -> list[int]:
    try:
        net_arch = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--policy-net-arch must be a comma-separated list of integers"
        ) from exc
    if not net_arch:
        raise argparse.ArgumentTypeError("--policy-net-arch cannot be empty")
    return net_arch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate SBX/JAX DQN and C-DQN against the RM-discrete SB3 baselines."
        )
    )
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--exploration-fraction", type=float)
    parser.add_argument("--exploration-final-eps", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2_500)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--policy-net-arch", type=_parse_net_arch, default=[64, 64])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tensorboard-log", type=Path, default=LOG_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--tb-log-suffix",
        help="Optional suffix appended to SBX TensorBoard run names.",
    )
    parser.add_argument("--baseline-dqn-log", type=Path, default=BASELINE_LOGS["dqn"])
    parser.add_argument(
        "--baseline-cdqn-log",
        type=Path,
        default=BASELINE_LOGS["cdqn"],
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training and compare existing SBX TensorBoard logs.",
    )
    parser.add_argument(
        "--sbx-dqn-log",
        type=Path,
        help="Existing SBX/JAX DQN TensorBoard directory for --plot-only.",
    )
    parser.add_argument(
        "--sbx-cdqn-log",
        type=Path,
        help="Existing SBX/JAX C-DQN TensorBoard directory for --plot-only.",
    )
    return parser.parse_args()


def main() -> None:
    """Run SBX validation training or compare existing TensorBoard logs."""
    args = parse_args()
    algorithms = _expand_algorithms(args.algorithms)
    baseline_logs = {
        "dqn": args.baseline_dqn_log,
        "cdqn": args.baseline_cdqn_log,
    }

    sbx_runs = []
    if args.plot_only:
        existing_logs = {
            "dqn": args.sbx_dqn_log,
            "cdqn": args.sbx_cdqn_log,
        }
        for algorithm in algorithms:
            log_dir = existing_logs[algorithm]
            if log_dir is None:
                raise ValueError(
                    f"--plot-only requires --sbx-{algorithm}-log for {algorithm}"
                )
            sbx_runs.append(
                {
                    "algorithm": algorithm,
                    "backend": "sbx",
                    "tensorboard_log": str(log_dir),
                }
            )
    else:
        for algorithm in algorithms:
            sbx_runs.append(
                run_sbx_algorithm(
                    algorithm=algorithm,
                    seed=args.seed,
                    total_timesteps=args.total_timesteps,
                    max_steps=args.max_steps,
                    learning_rate=args.learning_rate,
                    exploration_fraction=args.exploration_fraction,
                    exploration_final_eps=args.exploration_final_eps,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    learning_starts=args.learning_starts,
                    train_freq=args.train_freq,
                    gradient_steps=args.gradient_steps,
                    target_update_interval=args.target_update_interval,
                    policy_net_arch=args.policy_net_arch,
                    device=args.device,
                    tensorboard_log=args.tensorboard_log,
                    tb_log_suffix=args.tb_log_suffix,
                )
            )

    result = compare_runs(
        sbx_runs=sbx_runs,
        baseline_logs=baseline_logs,
        results_dir=args.results_dir,
    )

    print(f"Wrote comparison JSON to {result['artifacts']['json']}")
    print(f"Wrote comparison plot to {result['artifacts']['plot']}")
    for row in result["summary"]:
        final_value = row["final_ep_rew_mean"]
        suffix = ""
        if row.get("final_delta_vs_sb3") is not None:
            suffix = f", delta_vs_sb3={row['final_delta_vs_sb3']:.3f}"
        final_text = "n/a" if final_value is None else f"{final_value:.3f}"
        print(
            f"{row['backend']} {row['algorithm']}: "
            f"final_step={row['final_step']}, final_ep_rew_mean={final_text}"
            f"{suffix}"
        )


if __name__ == "__main__":
    main()
