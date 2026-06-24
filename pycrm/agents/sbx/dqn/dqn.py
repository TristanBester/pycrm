# pyright: reportMissingTypeStubs=false

from functools import partial
from typing import Any

import optax

try:
    from sbx import DQN as SBXDQN
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SBX DQN requires the optional SBX dependency. "
        + "Install it with `pip install pyrewardmachines[sbx]`."
    ) from exc


def _clipped_adam(
    *,
    learning_rate: float,
    max_grad_norm: float,
    **kwargs: Any,
) -> optax.GradientTransformation:
    """Match SB3 DQN's gradient clipping before the Adam update."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate, **kwargs),
    )


class DQN(SBXDQN):
    """SBX DQN wrapper with Stable-Baselines3-compatible defaults."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SBX DQN with SB3's DQN optimization semantics."""
        max_grad_norm = float(kwargs.pop("max_grad_norm", 10.0))
        policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
        policy_kwargs.setdefault(
            "optimizer_class",
            partial(_clipped_adam, max_grad_norm=max_grad_norm),
        )

        kwargs["policy_kwargs"] = policy_kwargs
        kwargs.setdefault("learning_rate", 1e-4)
        kwargs.setdefault("batch_size", 32)
        kwargs.setdefault("exploration_fraction", 0.1)
        kwargs.setdefault("exploration_final_eps", 0.05)
        kwargs.setdefault("target_update_interval", 10_000)
        super().__init__(*args, **kwargs)
