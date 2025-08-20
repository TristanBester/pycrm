import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class PuckWorld(gym.Env):
    """Puck World environment."""

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self):
        """Constructor for the Puck World environment."""
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,))
        self.action_space = gym.spaces.Discrete(4)

        self.target_one_pos = np.array([0.0, 0.0])
        self.target_two_pos = np.array([0.0, 0.0])
        self.target_three_pos = np.array([0.0, 0.0])

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Set the position and velocity of the agent and adversary
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_vel = np.array([0.0, 0.0])
        self.adversary_pos = np.array([0.8, 0.8])

        # Randomly set the positions of the targets
        self._set_target_one_pos()
        self._set_target_two_pos()
        self._set_target_three_pos()
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        self._update_agent_position_velocity(action)

        self._update_adversary_position()
        self._update_target_one_position()
        self._update_target_two_position()
        self._update_target_three_position()

        return self._get_obs(), 0.0, False, False, {}

    def render(self) -> None:
        """Render the current environment with an arrow for velocity."""
        plt.gca().cla()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().add_artist(plt.Circle(tuple(self.agent_pos), 0.05, color="blue"))
        plt.gca().add_artist(
            plt.Circle(tuple(self.target_one_pos), 0.02, color="green")
        )
        plt.gca().add_artist(plt.Circle(tuple(self.target_two_pos), 0.02, color="red"))
        plt.gca().add_artist(
            plt.Circle(tuple(self.target_three_pos), 0.02, color="orange")
        )
        plt.gca().add_artist(plt.Circle(tuple(self.adversary_pos), 0.02, color="black"))

        plt.gcf().gca().add_artist(
            plt.Circle(tuple(self.target_one_pos), 0.07, color="green", alpha=0.5)
        )
        plt.gca().add_artist(
            plt.Circle(tuple(self.target_two_pos), 0.07, color="red", alpha=0.5)
        )
        plt.gca().add_artist(
            plt.Circle(tuple(self.target_three_pos), 0.07, color="orange", alpha=0.5)
        )
        plt.gca().add_artist(
            plt.Circle(tuple(self.adversary_pos), 0.12, color="black", alpha=0.5)
        )

        plt.arrow(
            self.agent_pos[0],
            self.agent_pos[1],
            self.agent_vel[0],
            self.agent_vel[1],
            color="blue",
        )
        plt.pause(0.1)

    def _update_agent_position_velocity(self, action) -> None:
        """Update agent position and velocity based on selected action."""
        if action == self.RIGHT:
            self.agent_vel[0] += 0.05
        elif action == self.LEFT:
            self.agent_vel[0] -= 0.05
        elif action == self.UP:
            self.agent_vel[1] += 0.05
        elif action == self.DOWN:
            self.agent_vel[1] -= 0.05

        self.agent_vel += np.random.normal(0, 0.01, size=2)
        self.agent_vel = np.clip(self.agent_vel, -0.15, 0.15)
        self.agent_pos += self.agent_vel

        # Agent wraps around map if it goes out of bounds
        if self.agent_pos[0] < -1.0:
            self.agent_pos[0] = 1.0
        elif self.agent_pos[0] > 1.0:
            self.agent_pos[0] = -1.0
        if self.agent_pos[1] < -1.0:
            self.agent_pos[1] = 1.0
        elif self.agent_pos[1] > 1.0:
            self.agent_pos[1] = -1.0

    def _update_adversary_position(self) -> None:
        """Update adversary position based on agent position."""
        dx = self.agent_pos[0] - self.adversary_pos[0]
        dy = self.agent_pos[1] - self.adversary_pos[1]
        self.adversary_pos[0] += np.sign(dx) * 0.0005
        self.adversary_pos[1] += np.sign(dy) * 0.0005

    def _update_target_one_position(self) -> None:
        """Update target one position if agent gets close enough."""
        if np.linalg.norm(self.agent_pos - self.target_one_pos) < 0.15:
            self._set_target_one_pos()

    def _update_target_two_position(self) -> None:
        """Update target two position if agent gets close enough."""
        if np.linalg.norm(self.agent_pos - self.target_two_pos) < 0.15:
            self._set_target_two_pos()

    def _update_target_three_position(self) -> None:
        """Update target three position if agent gets close enough."""
        if np.linalg.norm(self.agent_pos - self.target_three_pos) < 0.15:
            self._set_target_three_pos()

    def _get_obs(self):
        """Get the current observation from environment state."""
        return np.concatenate(
            (
                self.agent_pos,
                self.agent_vel,
                self.target_one_pos,
                self.target_two_pos,
                self.target_three_pos,
                self.adversary_pos,
            ),
            dtype=np.float32,
        )

    def _set_target_one_pos(self):
        """Set target one position randomly."""
        while True:
            self.target_one_pos = np.random.uniform(-1.0, 1.0, size=2)

            # Ensure target one is not too close to the other targets
            dist_one = np.linalg.norm(self.target_one_pos - self.target_two_pos)
            dist_two = np.linalg.norm(self.target_one_pos - self.target_three_pos)

            if dist_one > 0.45 and dist_two > 0.45:
                break

    def _set_target_two_pos(self):
        """Set target two position randomly."""
        while True:
            self.target_two_pos = np.random.uniform(-1, 1, size=2)

            # Ensure target two is not too close to the other targets
            dist_one = np.linalg.norm(self.target_two_pos - self.target_one_pos)
            dist_two = np.linalg.norm(self.target_two_pos - self.target_three_pos)

            if dist_one > 0.45 and dist_two > 0.45:
                break

    def _set_target_three_pos(self):
        """Set target three position randomly."""
        while True:
            self.target_three_pos = np.random.uniform(-1, 1, size=2)

            # Ensure target three is not too close to the other targets
            dist_one = np.linalg.norm(self.target_three_pos - self.target_one_pos)
            dist_two = np.linalg.norm(self.target_three_pos - self.target_two_pos)

            if dist_one > 0.45 and dist_two > 0.45:
                break


if __name__ == "__main__":
    env = PuckWorld()
    env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(obs, reward, done, info)
