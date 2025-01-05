import os

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "../logs"
OUTPUT_DIR = "../outputs"


def main() -> None:
    """Main function."""
    rm_complexity = np.load(os.path.join(LOG_PATH, "implementation/rm_complexity.npy"))
    crm_complexity = np.load(
        os.path.join(LOG_PATH, "implementation/crm_complexity.npy")
    )

    plt.plot(range(1, len(rm_complexity) + 1), rm_complexity, label="RM")
    plt.plot(range(1, len(crm_complexity) + 1), crm_complexity, label="CRM (Ours)")
    plt.xlabel("Task Complexity")
    plt.ylabel("Cyclomatic Complexity")
    plt.yscale("log")
    plt.title("Implementation Complexity Comparison")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "implementation.png"), dpi=300)


if __name__ == "__main__":
    raise SystemExit(main())
