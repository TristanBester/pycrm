import os

import matplotlib.pyplot as plt
import pandas as pd

from experiments.warehouse.results.utils.plot import compute_results_with_ci

PARSED_LOG_PATH = "../outputs/parsed_logs.csv"
OUTPUT_DIR = "../outputs"


def main():
    df = pd.read_csv(PARSED_LOG_PATH)

    x_values, csac_mean, csac_lower, csac_upper, sac_mean, sac_lower, sac_upper = (
        compute_results_with_ci(df, "subtask/blue/complete_1", 13_500_000)
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, "subtask_blue_1.png")

    plt.plot(x_values, csac_mean, label="CSAC")
    plt.fill_between(x_values, csac_lower, csac_upper, alpha=0.2, label="99% CI")
    plt.plot(x_values, sac_mean, label="SAC")
    plt.fill_between(x_values, sac_lower, sac_upper, alpha=0.2, label="99% CI")

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Blue 1: CSAC vs SAC (AVG over 50 runs)")
    plt.savefig(output_path, dpi=300)
    # plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
