from functools import cached_property
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import wandb
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition


@chex.dataclass
class State:
    pos: chex.Array
    step_count: chex.Array
    key: chex.PRNGKey


class Observation(NamedTuple):
    agent_pos: chex.Array


class GridWorld(Environment[State, specs.DiscreteArray, Observation]):
    def __init__(self, grid_size: int = 5, max_steps: int = 100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_deltas = jnp.array(
            [
                (0, 1),  # right
                (0, -1),  # left
                (-1, 0),  # up
                (1, 0),  # down
            ]
        )
        self.a_pos = jnp.array([self.grid_size, -1, 0], dtype=jnp.int32)
        self.b_pos = jnp.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=jnp.int32
        )
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        state = State(
            pos=jnp.array([0, 0], dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        timestep = restart(observation=self._observation(state))
        return state, timestep

    def step(
        self, state: State, action: chex.Numeric
    ) -> tuple[State, TimeStep[Observation]]:
        delta = self.action_deltas[action]
        new_pos = state.pos + delta
        new_pos = jnp.clip(new_pos, 0, self.grid_size - 1)

        new_state = State(
            pos=new_pos,
            step_count=state.step_count + 1,
            key=state.key,
        )
        obs = self._observation(new_state)

        # goal = jnp.array([self.grid_size - 1, self.grid_size - 1], dtype=jnp.int32)
        # reached_goal = jnp.all(new_pos == goal)
        # reward = jnp.where(reached_goal, 1.0, -0.01).astype(jnp.float32)
        reached_goal = False
        reward = jnp.array(0.0, dtype=jnp.float32)

        time_out = new_state.step_count >= self.max_steps
        done = jnp.logical_or(reached_goal, time_out)

        timestep = jax.lax.cond(
            done,
            lambda: termination(reward=reward, observation=obs),
            lambda: transition(reward=reward, observation=obs),
        )
        return new_state, timestep

    def _observation(self, state: State) -> Observation:
        pos = state.pos.astype(jnp.float32)
        return Observation(agent_pos=pos)

    def render(self, state: State, *, print_result: bool = True) -> str:
        """Render a state as terminal-friendly ASCII art."""
        agent_row, agent_col = map(int, state.pos.tolist())
        horizontal = "+" + "---+" * self.grid_size
        rows = [
            f"GridWorld | step {int(state.step_count)} / {self.max_steps}",
            horizontal,
        ]

        for row in range(self.grid_size):
            cells: list[str] = []
            for col in range(self.grid_size):
                if (row, col) == (agent_row, agent_col):
                    cells.append(" X ")
                elif (row, col) == (self.grid_size - 1, 0):
                    cells.append(" A ")
                elif (row, col) == (self.grid_size - 1, self.grid_size - 1):
                    cells.append(" B ")
                else:
                    cells.append(" . ")
            rows.append("|" + "|".join(cells) + "|")
            rows.append(horizontal)

        rendered = "\n".join(rows)
        if print_result:
            print(rendered)
        return rendered

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        agent_pos = specs.BoundedArray(
            shape=(2,),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=float(self.grid_size - 1),
            name="agent_pos",
        )
        return specs.Spec(Observation, "ObservationSpace", agent_pos=agent_pos)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=4, name="action")
