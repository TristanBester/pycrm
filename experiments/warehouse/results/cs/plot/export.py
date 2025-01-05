import os

import pandas as pd

from experiments.warehouse.results.utils import compute_results_with_ci

PARSED_LOG_PATH = "../outputs/parsed_logs.csv"
OUTPUT_DIR = "../outputs"


def main() -> None:
    """Main function."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(PARSED_LOG_PATH)

    (
        x_values_return,
        csac_mean_return,
        csac_lower_return,
        csac_upper_return,
        sac_mean_return,
        sac_lower_return,
        sac_upper_return,
    ) = compute_results_with_ci(df, "rollout/ep_rew_mean", 10_000_000)
    (
        x_values_success,
        csac_mean_success,
        csac_lower_success,
        csac_upper_success,
        sac_mean_success,
        sac_lower_success,
        sac_upper_success,
    ) = compute_results_with_ci(df, "subtask/blue/complete_1", 10_000_000)

    # Take the values from each array at every 1000 steps
    x_values_return = x_values_return[::100]
    csac_mean_return = csac_mean_return[::100]
    csac_lower_return = csac_lower_return[::100]
    csac_upper_return = csac_upper_return[::100]
    sac_mean_return = sac_mean_return[::100]
    sac_lower_return = sac_lower_return[::100]
    sac_upper_return = sac_upper_return[::100]

    x_values_success = x_values_success[::100]
    csac_mean_success = csac_mean_success[::100]
    csac_lower_success = csac_lower_success[::100]
    csac_upper_success = csac_upper_success[::100]
    sac_mean_success = sac_mean_success[::100]
    sac_lower_success = sac_lower_success[::100]
    sac_upper_success = sac_upper_success[::100]

    data = {
        "samples_return": x_values_return,
        "csac_mean_return": csac_mean_return,
        "csac_lower_return": csac_lower_return,
        "csac_upper_return": csac_upper_return,
        "sac_mean_return": sac_mean_return,
        "sac_lower_return": sac_lower_return,
        "sac_upper_return": sac_upper_return,
        "samples_success": x_values_success,
        "csac_mean_success": csac_mean_success,
        "csac_lower_success": csac_lower_success,
        "csac_upper_success": csac_upper_success,
        "sac_mean_success": sac_mean_success,
        "sac_lower_success": sac_lower_success,
        "sac_upper_success": sac_upper_success,
    }
    pd.DataFrame(data).to_csv(
        os.path.join(OUTPUT_DIR, "context-sensitive.csv"), index=False
    )


if __name__ == "__main__":
    raise SystemExit(main())
