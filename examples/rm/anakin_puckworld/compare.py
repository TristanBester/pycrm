# ruff: noqa: D103,E501,B905
"""Run SB3-DQN and Anakin-DQN on PuckWorld and compare speed + convergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.rm.anakin_puckworld.train_sb3 import run_sb3_dqn
from examples.rm.anakin_puckworld.train_stoix_anakin import run_anakin_dqn

SOLVED_THRESHOLD = 0.0  # mean episode return that counts as "task solved"; tune per run


def _time_to_threshold(steps, wall, returns, threshold):
    for s, w, r in zip(steps, wall, returns):
        if r >= threshold:
            return s, w
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    runs = [
        run_sb3_dqn(args.timesteps, args.seed, args.out),
        run_anakin_dqn(args.timesteps, args.seed, args.out),
    ]

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Learning curves.
    for xkey, fname, xlabel in [
        ("steps", "return_vs_steps.png", "environment steps"),
        ("wall_clock", "return_vs_walltime.png", "wall-clock seconds"),
    ]:
        plt.figure()
        for m in runs:
            plt.plot(m[xkey], m["returns"], label=m["backend"])
        plt.xlabel(xlabel)
        plt.ylabel("episode return")
        plt.legend()
        plt.title("PuckWorld: Anakin vs SB3")
        plt.savefig(Path(args.out) / fname, dpi=120, bbox_inches="tight")
        plt.close()

    # Results table + CSV.
    header = f"{'backend':<22}{'steps/s':>12}{'wall(s)':>10}{'final_ret':>12}{'steps@thr':>12}{'wall@thr':>12}"
    print(header)
    csv = Path(args.out) / "comparison.csv"
    with csv.open("w") as fh:
        fh.write("backend,steps_per_sec,total_wall_clock,final_return,steps_to_threshold,wall_to_threshold\n")
        for m in runs:
            final = m["returns"][-1] if m["returns"] else float("nan")
            s_thr, w_thr = _time_to_threshold(
                m["steps"], m["wall_clock"], m["returns"], SOLVED_THRESHOLD
            )
            print(f"{m['backend']:<22}{m['steps_per_sec']:>12.0f}"
                  f"{m['total_wall_clock']:>10.1f}{final:>12.1f}"
                  f"{str(s_thr):>12}{('%.1f' % w_thr) if w_thr else 'n/a':>12}")
            fh.write(f"{m['backend']},{m['steps_per_sec']:.1f},"
                     f"{m['total_wall_clock']:.3f},{final:.3f},"
                     f"{s_thr},{w_thr}\n")


if __name__ == "__main__":
    main()
