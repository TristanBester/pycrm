import os

import numpy as np
import pandas as pd

LOG_PATH = "../logs"
OUTPUT_DIR = "../outputs"


def main() -> None:
    """Main function."""
    rm_edge_count = np.load(os.path.join(LOG_PATH, "theoretical/rm_edge_count.npy"))
    rm_node_count = np.load(os.path.join(LOG_PATH, "theoretical/rm_node_count.npy"))
    rm_complexity = np.load(os.path.join(LOG_PATH, "implementation/rm_complexity.npy"))
    crm_edge_count = np.load(os.path.join(LOG_PATH, "theoretical/crm_edge_count.npy"))
    crm_node_count = np.load(os.path.join(LOG_PATH, "theoretical/crm_node_count.npy"))
    crm_complexity = np.load(
        os.path.join(LOG_PATH, "implementation/crm_complexity.npy")
    )
    n = list(range(1, len(rm_edge_count) + 1))

    df = pd.DataFrame(
        {
            "x": n,
            "rm_edge": rm_edge_count,
            "rm_node": rm_node_count,
            "rm_complexity": rm_complexity,
            "crm_edge": crm_edge_count,
            "crm_node": crm_node_count,
            "crm_complexity": crm_complexity,
        }
    )
    df.to_csv(os.path.join(OUTPUT_DIR, "complexity-analysis.csv"), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
