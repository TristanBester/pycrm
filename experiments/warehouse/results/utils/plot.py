from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.warehouse.results.utils.ci import get_bootstrap_ci_for_mean


def compute_results_with_ci(df, statistic, max_steps, alpha=0.01):
    algo_results = _load_all_results_for_statistic(df, statistic, max_steps)

    x_values = np.arange(0, max_steps, 1_000)
    print(f"Computing CSAC CI...")
    csac_mean, csac_lower, csac_upper = get_bootstrap_ci_for_mean(
        algo_results["CSAC"],
        10000,
        alpha,  # type: ignore
    )
    print(f"Computing SAC CI...")
    sac_mean, sac_lower, sac_upper = get_bootstrap_ci_for_mean(
        algo_results["SAC"],
        10000,
        alpha,  # type: ignore
    )

    return x_values, csac_mean, csac_lower, csac_upper, sac_mean, sac_lower, sac_upper


def _load_all_results_for_statistic(df: pd.DataFrame, statistic: str, max_steps: int):
    algo_seed_combinations = []
    for _, r in df[["algorithm", "seed"]].drop_duplicates().iterrows():
        algo_seed_combinations.append((r.algorithm, r.seed))

    algo_results = defaultdict(list)
    print(f"Loading results...")
    for algo, seed in tqdm(algo_seed_combinations):
        # Load rowws for this algo and seed
        df_algo_seed = df[(df.algorithm == algo) & (df.seed == seed)]
        df_algo_seed = df_algo_seed.sort_values(by="step")
        df_algo_seed = df_algo_seed[df_algo_seed.step < max_steps].copy()

        # get the values for the statistic
        df_statistic = df_algo_seed[df_algo_seed.tag == statistic]
        y_values = df_statistic.value.values
        x_values = df_statistic.step.values

        # remove failed runs
        if len(y_values) < 100:
            print(f"Skipping {algo} {seed} because it has less than 100 runs")
            continue

        # force all runs to have the same length
        ys = _standardize_length(x_values, y_values, 0, max_steps, 1_000)
        algo_results[algo].append(ys)

    for algo, results in algo_results.items():
        algo_results[algo] = np.array(results)  # type: ignore
    return algo_results


def _standardize_length(x_values, y_values, x_min, x_max, step):
    df_result = pd.DataFrame(
        {
            "x": x_values,
            "y": y_values,
        }
    )
    results = df_result.groupby("x").y.mean()
    x_values = np.array(results.index)
    y_values = np.array(results.values)

    xs = np.arange(x_min, x_max, step)
    ys = np.interp(xs, x_values, y_values)
    return ys
