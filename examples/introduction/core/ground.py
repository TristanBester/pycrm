import gymnasium as gym
import numpy as np


class LetterWorld(gym.Env):
    """The Letter World environment."""

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    A_POSITION = np.array([1, 1])
    C_POSITION = np.array([1, 5])
    N_ROWS = 3
    N_COLS = 7

    def __init__(self) -> None:
        """Initialize the environment."""
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(3,), dtype=np.int32
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        self.symbol_seen = False
        self.agent_position = np.array([1, 3])
        return self._get_obs(), {}

    def _get_obs(self):
        """Return the observation of the agent."""
        symbol_seen = np.array([1]) if self.symbol_seen else np.array([0])
        return np.concatenate([symbol_seen, self.agent_position])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        self._update_agent_position(action)

        if not self.symbol_seen and np.array_equal(
            self.agent_position, self.A_POSITION
        ):
            self.symbol_seen = np.random.random() < 0.5
        return self._get_obs(), 0, False, False, {}

    def _update_agent_position(self, action: int) -> None:
        """Moves that take agent out of the grid leave it in the same position."""
        row, col = self.agent_position

        if action == self.RIGHT:
            n_row = row
            n_col = col + 1 if col < self.N_COLS - 1 else col
        elif action == self.LEFT:
            n_row = row
            n_col = col - 1 if col > 0 else col
        elif action == self.UP:
            n_col = col
            n_row = row - 1 if row > 0 else row
        elif action == self.DOWN:
            n_col = col
            n_row = row + 1 if row < self.N_ROWS - 1 else row
        else:
            raise ValueError(f"Invalid action {action}.")
        self.agent_position = (n_row, n_col)

    def render(self) -> None:
        """Render the environment as a string."""
        str_repr = "+" + "-" * (self.N_COLS * 2 - 1) + "+\n"  # Top border

        for r in range(self.N_ROWS):
            str_repr += "|"  # Left border
            for c in range(self.N_COLS):
                if np.array_equal(np.array([r, c]), self.agent_position):
                    str_repr += "\x1b[1;37;42m" + "x" + "\x1b[0m" + " "
                elif (
                    np.array_equal(np.array([r, c]), self.A_POSITION)
                    and not self.symbol_seen
                ):
                    str_repr += "\x1b[1;37;44m" + "A" + "\x1b[0m" + " "
                elif (
                    np.array_equal(np.array([r, c]), self.A_POSITION)
                    and self.symbol_seen
                ):
                    str_repr += "\x1b[1;37;44m" + "B" + "\x1b[0m" + " "
                elif np.array_equal(np.array([r, c]), self.C_POSITION):
                    str_repr += "\x1b[1;37;44m" + "C" + "\x1b[0m" + " "
                else:
                    str_repr += "." + " "
            str_repr = str_repr.rstrip() + "|\n"  # Right border, remove trailing space

        str_repr += "+" + "-" * (self.N_COLS * 2 - 1) + "+"  # Bottom border
        print(str_repr)
