import gymnasium as gym
import numpy as np


class OfficeWorld(gym.Env):
    """OfficeWorld environment."""

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self, max_n: int = 2, start_row: int = 3, start_col: int = 11) -> None:
        """Contructor for the OfficeWorld environment."""
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)

        self.max_n = max_n
        self.start_row = start_row
        self.start_col = start_col

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self._setup_grid()

        self.total_mail = np.random.randint(1, self.max_n + 1)
        self.mail_collected = 0

        self.agent_row = self.start_row
        self.agent_col = self.start_col
        self.state = np.array([self.agent_row, self.agent_col, 0])

        return self.state, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        self._update_agent_position(action)

        # Determine if all mail has been collected
        if self.state[0] == 6 and self.state[1] == 10:
            if self.mail_collected < self.total_mail:
                self.mail_collected += 1
            else:
                self.state[2] = 1
        return self.state, 0, False, False, {}

    def render(self) -> None:
        """Render the environment."""
        for i in range(13):
            for j in range(17):
                if (i, j) == (self.state[0], self.state[1]):
                    print("\x1b[5;30;42m" + "X" + "\x1b[0m", end=" ")
                elif self.grid[i, j] != 0:
                    if (i, j) == (6, 10) and self.state[2] == 1:
                        obj = "E"
                    else:
                        obj = self.grid_values[self.grid[i, j]]
                    print(obj, end=" ")
                else:
                    print(".", end=" ")
            print()

    def _setup_grid(self) -> None:
        """Setup the grid for the environment."""
        self.grid = np.zeros((13, 17))
        self.grid_values = {0: "", 1: "X", 2: "*", 3: "P", 4: "M", 5: "C", 6: "E"}

        # Add walls
        self.grid[0, :] = 1
        self.grid[12, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, 16] = 1
        self.grid[4, 1] = 1
        self.grid[4, 15] = 1
        self.grid[4, 3:6] = 1
        self.grid[4, 7:10] = 1
        self.grid[4, 11:14] = 1
        self.grid[8, 1] = 1
        self.grid[8, 15] = 1
        self.grid[8, 3:14] = 1

        for r in range(1, 12):
            if r in (2, 10):
                continue

            for c in range(17):
                if c in (4, 8, 12):
                    self.grid[r, c] = 1

        # Add decorations
        self.grid[6, [2, 14]] = 2
        self.grid[10, [6, 10]] = 2

        # Add people
        self.grid[6, 6] = 3
        # Add mail
        self.grid[6, 10] = 4
        # Add coffee
        self.grid[2, 6] = 5

    def _update_agent_position(self, action) -> None:
        """Update the agent position based on the selected action."""
        if action == self.RIGHT:
            new_agent_pos = (self.state[0], self.state[1] + 1)
        elif action == self.LEFT:
            new_agent_pos = (self.state[0], self.state[1] - 1)
        elif action == self.UP:
            new_agent_pos = (self.state[0] - 1, self.state[1])
        elif action == self.DOWN:
            new_agent_pos = (self.state[0] + 1, self.state[1])
        else:
            raise ValueError(f"Invalid action {action}.")

        # Handle collisions with walls
        if self.grid[new_agent_pos[0], new_agent_pos[1]] != 1:
            self.state[0] = new_agent_pos[0]
            self.state[1] = new_agent_pos[1]
