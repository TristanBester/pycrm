import os

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from experiments.complexity.lib.automata import (
    compute_crm_transition_graph,
    compute_rm_transition_graph,
)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    rm_node_count = np.zeros(config.exp.max_n)
    rm_edge_counts = np.zeros(config.exp.max_n)
    crm_node_count = np.zeros(config.exp.max_n)
    crm_edge_counts = np.zeros(config.exp.max_n)

    for n in trange(1, config.exp.max_n + 1):
        rm = compute_rm_transition_graph(n)
        rm_node_count[n - 1] = len(rm.nodes)
        rm_edge_counts[n - 1] = len(rm.edges)

        crm = compute_crm_transition_graph()
        crm_node_count[n - 1] = len(crm.nodes)
        crm_edge_counts[n - 1] = len(crm.edges)

    output_dir = os.path.join(config.exp.log_dir, "theoretical")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, "rm_node_count.npy"), rm_node_count)
    np.save(os.path.join(output_dir, "rm_edge_count.npy"), rm_edge_counts)
    np.save(os.path.join(output_dir, "crm_node_count.npy"), crm_node_count)
    np.save(os.path.join(output_dir, "crm_edge_count.npy"), crm_edge_counts)


if __name__ == "__main__":
    main()
