# ruff: noqa: D103, E501
"""Run SB3-DQN and Anakin-DQN on LetterWorld across seeds; compare time-to-solve."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.rm.letterworld_anakin.train_anakin import run_anakin_dqn
from examples.rm.letterworld_anakin.train_sb3 import run_sb3_dqn

SOLVE = 0.95


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/letterworld")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = {"anakin_dqn": run_anakin_dqn, "sb3_dqn": run_sb3_dqn}
    runs = {name: [] for name in arms}
    for name, fn in arms.items():
        for seed in range(args.seeds):
            runs[name].append(fn(args.timesteps, seed, str(out)))

    # success-rate vs wall-clock and vs steps; return vs wall-clock
    for xkey, ykey, fname, xlabel, ylabel in [
        ("eval_wall", "success_rate", "success_vs_walltime.png", "wall-clock (s)", "success rate"),
        ("eval_steps", "success_rate", "success_vs_steps.png", "env steps", "success rate"),
        ("eval_wall", "mean_return", "return_vs_walltime.png", "wall-clock (s)", "mean return"),
    ]:
        plt.figure()
        colors = {"anakin_dqn": "#2563eb", "sb3_dqn": "#dc2626"}
        for name in arms:
            for i, m in enumerate(runs[name]):
                plt.plot(m[xkey], m[ykey], color=colors[name], alpha=0.25, linewidth=1,
                         label=name if i == 0 else None)
        if ykey == "success_rate":
            plt.axhline(SOLVE, color="gray", linestyle="--", linewidth=1, label=f"solved={SOLVE}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f"LetterWorld: {ylabel} vs {xlabel}")
        plt.savefig(out / fname, dpi=120, bbox_inches="tight")
        plt.close()

    header = f"{'backend':<14}{'solved/seeds':>14}{'t2solve mean±std(s)':>24}{'steps2solve mean±std':>26}"
    print(header)
    csv = out / "comparison.csv"
    with csv.open("w") as fh:
        fh.write("backend,solved,seeds,time_to_solve_mean,time_to_solve_std,steps_to_solve_mean,steps_to_solve_std\n")
        for name in arms:
            tmean, tstd, nsolved = _agg([m["time_to_solve"] for m in runs[name]])
            smean, sstd, _ = _agg([m["steps_to_solve"] for m in runs[name]])
            tcell = "n/a" if tmean is None else f"{tmean:.1f}±{tstd:.1f}"
            scell = "n/a" if smean is None else f"{smean:.0f}±{sstd:.0f}"
            print(f"{name:<14}{f'{nsolved}/{args.seeds}':>14}{tcell:>24}{scell:>26}")
            fh.write(f"{name},{nsolved},{args.seeds},{tmean},{tstd},{smean},{sstd}\n")


if __name__ == "__main__":
    main()
