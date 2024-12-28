import os

import pandas as pd
from tbparse import SummaryReader
from tqdm import tqdm

LOG_DIR = "/Users/tristan/Projects/counting-reward-machines/experiments/warehouse/results/cs/logs"


def main() -> None:
    all_dfs = []
    for model_name in tqdm(os.listdir(LOG_DIR)):
        log_path = os.path.join(LOG_DIR, model_name)
        reader = SummaryReader(log_path)
        df = reader.scalars
        df["model_name"] = model_name
        df["algorithm"] = model_name.split("_")[0]
        df["seed"] = model_name.split("_")[2]
        all_dfs.append(df)

    log_df = pd.concat(all_dfs)
    log_df.to_csv("../outputs/parsed_logs.csv")


if __name__ == "__main__":
    raise SystemExit(main())
