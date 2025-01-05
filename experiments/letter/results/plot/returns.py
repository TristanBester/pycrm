import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.letter.results.utils import get_bootstrap_ci_for_mean

LOG_PATH = "../logs"
OUTPUT_DIR = "../outputs"


def main() -> None:
    ql_results = np.load(os.path.join(LOG_PATH, "results_ql.npy"))
    cql_results = np.load(os.path.join(LOG_PATH, "results_cql.npy"))

    ql_mean, ql_lower, ql_upper = get_bootstrap_ci_for_mean(ql_results, alpha=0.1)
    cql_mean, cql_lower, cql_upper = get_bootstrap_ci_for_mean(cql_results, alpha=0.1)

    ql_mean = np.convolve(ql_mean, np.ones(100) / 100, mode="valid")[:2000]
    cql_mean = np.convolve(cql_mean, np.ones(100) / 100, mode="valid")[:2000]
    ql_lower = np.convolve(ql_lower, np.ones(100) / 100, mode="valid")[:2000]
    cql_lower = np.convolve(cql_lower, np.ones(100) / 100, mode="valid")[:2000]
    ql_upper = np.convolve(ql_upper, np.ones(100) / 100, mode="valid")[:2000]
    cql_upper = np.convolve(cql_upper, np.ones(100) / 100, mode="valid")[:2000]

    x_values = np.arange(0, ql_mean.shape[0])

    plt.plot(ql_mean, label="QL")
    plt.fill_between(x_values, ql_lower, ql_upper, alpha=0.2)
    plt.plot(cql_mean, label="CQL")
    plt.fill_between(x_values, cql_lower, cql_upper, alpha=0.2)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.title("Returns: QL vs CQL (AVG over 20 runs)")
    plt.savefig(os.path.join(OUTPUT_DIR, "returns.png"), dpi=300)


if __name__ == "__main__":
    raise SystemExit(main())
