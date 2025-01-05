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

    data = {
        "samples": x_values,
        "ql_mean": ql_mean,
        "ql_lower": ql_lower,
        "ql_upper": ql_upper,
        "cql_mean": cql_mean,
        "cql_lower": cql_lower,
        "cql_upper": cql_upper,
    }
    pd.DataFrame(data).to_csv(os.path.join(OUTPUT_DIR, "letter.csv"), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
