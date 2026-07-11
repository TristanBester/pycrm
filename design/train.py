import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from design.ground import GridWorld


def main():
    env = GridWorld(grid_size=5, max_steps=100)
    key = jax.random.PRNGKey(0)

    state, timestep = env.reset(key)
    print(
        f"type={timestep.step_type} pos={timestep.observation.agent_pos} reward={timestep.reward} discount={timestep.discount}"
    )
    env.render(state)

    for _ in range(100):
        action = int(input("action: "))
        state, timestep = env.step(state, action)
        print(
            f"type={timestep.step_type} pos={timestep.observation.agent_pos} reward={timestep.reward} discount={timestep.discount}"
        )
        env.render(state)

        if timestep.last():
            break


if __name__ == "__main__":
    main()
