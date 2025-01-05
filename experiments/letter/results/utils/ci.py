import numpy as np
from tqdm import trange


def get_bootstrap_ci_for_mean(
    dataset: np.ndarray, n_samples: int = 10000, alpha=0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bootstrap_estimates = _get_bootstrap_mean_estimates(dataset, n_samples)
    mean = np.mean(dataset, axis=0)
    quatiles = np.quantile(bootstrap_estimates, [alpha, 1 - alpha], axis=0)
    return mean, quatiles[0], quatiles[1]


def _get_bootstrap_mean_estimates(
    dataset: np.ndarray, n_samples: int = 10000
) -> np.ndarray:
    means = []

    for _ in trange(n_samples):
        sample_idx = np.random.choice(
            np.arange(len(dataset)), size=len(dataset), replace=True
        )
        samples = dataset[sample_idx]
        mean_estimate = np.mean(samples, axis=0)
        means.append(mean_estimate)
    return np.array(means)
