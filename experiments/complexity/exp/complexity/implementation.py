import os

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from experiments.complexity.lib.automata import (
    compute_crm_transition_graph,
    compute_rm_transition_graph,
)
from experiments.complexity.lib.complexity.cyclomatic import cyclomatic_complexity


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    rm_complexity = np.zeros(config.exp.max_n)
    crm_complexity = np.zeros(config.exp.max_n)

    for n in trange(1, config.exp.max_n + 1):
        rm = compute_rm_transition_graph(n)
        rm_complexity[n - 1] = cyclomatic_complexity(rm)

        crm = compute_crm_transition_graph()
        crm_complexity[n - 1] = cyclomatic_complexity(crm)

    output_dir = os.path.join(config.exp.log_dir, "implementation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, "rm_complexity.npy"), rm_complexity)
    np.save(os.path.join(output_dir, "crm_complexity.npy"), crm_complexity)


if __name__ == "__main__":
    main()
