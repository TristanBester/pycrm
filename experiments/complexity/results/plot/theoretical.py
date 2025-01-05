import os

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "../logs"
OUTPUT_DIR = "../outputs"


def main() -> None:
    """Main function."""
    rm_node_count = np.load(os.path.join(LOG_PATH, "theoretical/rm_node_count.npy"))
    rm_edge_counts = np.load(os.path.join(LOG_PATH, "theoretical/rm_edge_count.npy"))
    crm_node_count = np.load(os.path.join(LOG_PATH, "theoretical/crm_node_count.npy"))
    crm_edge_counts = np.load(os.path.join(LOG_PATH, "theoretical/crm_edge_count.npy"))

    plt.plot(range(1, len(rm_node_count) + 1), rm_node_count, label="RM Vertex Count")
    plt.plot(range(1, len(rm_edge_counts) + 1), rm_edge_counts, label="RM Edge Count")
    plt.plot(
        range(1, len(crm_node_count) + 1), crm_node_count, label="CRM Vertex Count"
    )
    plt.plot(
        range(1, len(crm_edge_counts) + 1), crm_edge_counts, label="CRM Edge Count"
    )
    plt.xlabel("Task Complexity")
    plt.ylabel("Machine Complexity")
    plt.title("Theoretical Complexity Comparison")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "theoretical.png"), dpi=300)


if __name__ == "__main__":
    raise SystemExit(main())
