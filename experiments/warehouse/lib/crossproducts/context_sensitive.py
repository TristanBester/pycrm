import time

import gymnasium as gym
import numpy as np

from crm.crossproduct.crossproduct import CrossProduct
from experiments.warehouse.lib.label.function import WarehouseLabellingFunction
from experiments.warehouse.lib.machines.context_sensitive import ContextSensitiveCRM

"""
NOTE: The ground obs here needs to be preprocessed to not be in the form of a dictionary
as in the old code we were always post processing the environment action with 
x =  self._ground_obs["observation"][:7]
"""


class ContextSensitiveCrossProductMDP(CrossProduct):
    def __init__(
        self, max_steps: int = 5000, render_mode="rgb_array", crm_kwargs: dict = {}
    ):
        # TODO: Pass all args for ground_env, labelling_function, crm through contrustor here to allow customization
        ground_env = gym.make("WarehouseGround-v0", render_mode=render_mode)
        super().__init__(
            ground_env=ground_env,  # type: ignore
            lf=WarehouseLabellingFunction(),
            crm=ContextSensitiveCRM(**crm_kwargs),
            max_steps=max_steps,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(4,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self._init_subtask_logging()

        # SUPER
        super().reset(seed=seed, options=options)
        self.steps = 0

        self.u = self.crm.u_0
        self.c = self.crm.c_0
        self.ground_obs, _ = self.ground_env.reset()

        self.ground_obs_next = self.ground_obs
        self.action = np.zeros(self.action_space.shape)  # type: ignore
        return self._get_obs(self.ground_obs, self.u, self.c), {}
        # END OF SUPER

    def step(self, action) -> tuple:
        action = self._preprocess_action(action)

        # SUPER OVERRIDE
        self.steps += 1

        # Update action
        self.last_action = self.action
        self.action = action

        # Update ground environment state
        self.ground_obs = self.ground_obs_next
        self.ground_obs_next, _, _, _, _ = self.ground_env.step(action)

        # Compute high-level events
        self.props = self.lf(self.ground_obs, action, self.ground_obs_next)  # type: ignore

        # Compute transition in cross-product MDP
        u_next, c_next, reward_fn = self.crm.transition(self.u, self.c, self.props)  # type: ignore
        # Compute transition reward
        reward = reward_fn(self.ground_obs, action, self.ground_obs_next)

        # Test if the episode is terminated
        terminated = u_next in self.crm.F
        truncated = self.steps >= self.max_steps

        self.u_last = self.u
        self.c_last = self.c
        self.reward = reward
        self.done = terminated or truncated
        self.u = u_next
        self.c = c_next

        obs, reward, terminated, truncated, info = (
            self._get_obs(self.ground_obs_next, self.u, self.c),
            reward,
            terminated,
            truncated,
            {},
        )
        # END OF SUPER
        # obs, reward, terminated, truncated, info = super().step(action)

        # Update waypoint visualisations
        # self.ground_env._env.task.update_waypoints(ee_pos=obs[:3])

        if self._machine_cfg_changed():
            c_o = tuple([int(c) for c in self.c])
            c_n = tuple([int(c) for c in c_next])
            print(f"U: {self.u} -> {u_next}, C: {c_o} -> {c_n}, Reward: {reward}")
            self._update_subtask_logging()
            self._update_block_positions()

        # FIXME: Move to ground env
        if self.last_action[-1] != self.action[-1]:
            self._handle_gripper_state_change()

        if terminated or truncated:
            info = self._get_subtask_logs()
        return obs, reward, terminated, truncated, info

    def _machine_cfg_changed(self):
        return self.u_last != self.u or self.c_last != self.c

    def render(self):
        pass

    def _get_obs(self, ground_obs, u, c):
        u_enc = self.crm.encode_machine_state(u=u)
        c_cfg_enc = self.crm.encode_counter_configuration(c=c, scale=100_000)
        c_state_enc = self.crm.encode_counter_state(c=c)
        return np.concatenate(
            [
                ground_obs,
                u_enc,
                c_cfg_enc,
                c_state_enc,
            ]
        )

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        0 / 0
        return obs[:7]

    # OVERRIDE
    def generate_counterfactual_experience(self, ground_obs, action, ground_obs_next):
        obs_buffer = []
        action_buffer = []
        obs_next_buffer = []
        reward_buffer = []
        done_buffer = []
        info_buffer = []

        for u_i in self.crm.U:
            for c_i in self.crm.sample_counter_configurations():
                # fixme: shouldn't require try except
                try:
                    u_j, c_j, rf_j = self.crm.transition(u_i, c_i, self.props)  # type: ignore
                except:
                    continue

                r_j = rf_j(ground_obs, action, ground_obs_next)

                obs_buffer.append(self._get_obs(ground_obs, u_i, c_i))
                action_buffer.append(action)
                obs_next_buffer.append(self._get_obs(ground_obs_next, u_j, c_j))
                reward_buffer.append(r_j)
                done_buffer.append(u_j in self.crm.F)
                info_buffer.append({})

        return (
            np.array(obs_buffer),
            np.array(action_buffer),
            np.array(obs_next_buffer),
            np.array(reward_buffer),
            np.array(done_buffer),
            np.array(info_buffer),
        )

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        # Clip end-effector displacement
        action[:3] = np.clip(action[:3], -0.1, 0.1)

        # Set gripper action
        if action[-1] <= 0:
            # Close gripper
            action[-1] = -1.0
        else:
            # Open gripper
            action[-1] = 1.0
        return action

    def _update_subtask_logging(self):
        # Update tracking variables
        if self.u == 0:
            # FIXME: The logging in this section assumes task strucure
            # For example with task (0,2,0), the blue block one is assumed to be placed
            # even though it has not

            # Test for completion of green blocks
            if self.c[0] == 2:
                self.red_three_complete = True
            elif self.c[0] == 1:
                self.red_two_complete = True
            elif self.c[0] == 0:
                self.red_one_complete = True

            # Test for completion of red blocks
            if self.c[1] == 2:
                self.green_three_complete = True
            elif self.c[1] == 1:
                self.green_two_complete = True
            elif self.c[1] == 0:
                self.green_one_complete = True

            # Test for completion of blue blocks
            if self.c[2] == 2:
                self.blue_three_complete = True
            elif self.c[2] == 1:
                self.blue_two_complete = True
            elif self.c[2] == 0:
                self.blue_one_complete = True

        elif self.u == 1:
            if self.c[2] == 3:
                self.blue_release_3_0 = True
            elif self.c[2] == 2:
                self.blue_release_2_0 = True
            elif self.c[2] == 1:
                self.blue_release_1_0 = True
        elif self.u == 2:
            if self.c[2] == 3:
                self.blue_grasp_3_0 = True
            elif self.c[2] == 2:
                self.blue_grasp_2_0 = True
            elif self.c[2] == 1:
                self.blue_grasp_1_0 = True
        elif self.u == 4:
            if self.c[2] == 3:
                self.blue_above_3_0 = True
            elif self.c[2] == 2:
                self.blue_above_2_0 = True
            elif self.c[2] == 1:
                self.blue_above_1_0 = True
        elif self.u == 6:
            if self.c[1] == 3:
                self.green_release_3_0 = True
            elif self.c[1] == 2:
                self.green_release_2_0 = True
            elif self.c[1] == 1:
                self.green_release_1_0 = True
        elif self.u == 7:
            if self.c[1] == 3:
                self.green_grasp_3_0 = True
            elif self.c[1] == 2:
                self.green_grasp_2_0 = True
            elif self.c[1] == 1:
                self.green_grasp_1_0 = True
        elif self.u == 9:
            if self.c[1] == 3:
                self.green_above_3_0 = True
            elif self.c[1] == 2:
                self.green_above_2_0 = True
            elif self.c[1] == 1:
                self.green_above_1_0 = True
        elif self.u == 11:
            if self.c[0] == 3:
                self.red_release_3_0 = True
            elif self.c[0] == 2:
                self.red_release_2_0 = True
            elif self.c[0] == 1:
                self.red_release_1_0 = True
        elif self.u == 12:
            if self.c[0] == 3:
                self.red_grasp_3_0 = True
            elif self.c[0] == 2:
                self.red_grasp_2_0 = True
            elif self.c[0] == 1:
                self.red_grasp_1_0 = True
        elif self.u == 14:
            if self.c[0] == 3:
                self.red_above_3_0 = True
            elif self.c[0] == 2:
                self.red_above_2_0 = True
            elif self.c[0] == 1:
                self.red_above_1_0 = True

    def _update_block_positions(self):
        # FIXME: Most of this logic should (probably) be moved to the ground environment
        # Reset block positions
        if self.u == 0 and self.c[0] != 0:
            if self.action[-1] <= 0:
                # Close gripper
                action = np.concatenate([np.zeros(3), np.array([-1.0])])
            else:
                # Open gripper
                action = np.concatenate([np.zeros(3), np.array([1.0])])

            for _ in range(10):
                self.ground_obs_next, _, _, _, _ = self.ground_env.step(action)

                if self.render_mode == "human":
                    ee_pos = self.ground_obs_next[:3]
                    # self.ground_env._env.task.update_waypoints(ee_pos=ee_pos)
                    self.render()
                    time.sleep(self.frame_delay)

            # self.ground_env._env.task.reset_red_block_pose()
        elif self.u == 0 and self.c[1] != 0:
            if self.action[-1] <= 0:
                # Close gripper
                action = np.concatenate([np.zeros(3), np.array([-1.0])])
            else:
                # Open gripper
                action = np.concatenate([np.zeros(3), np.array([1.0])])

            for _ in range(10):
                self.ground_obs_next, _, _, _, _ = self.ground_env.step(action)

                if self.render_mode == "human":
                    ee_pos = self.ground_obs_next[:3]
                    # self.ground_env._env.task.update_waypoints(ee_pos=ee_pos)
                    self.render()
                    time.sleep(self.frame_delay)

            # self.ground_env._env.task.reset_green_block_pose()
        elif self.u == 0 and self.c[2] != 0:
            if self.action[-1] <= 0:
                # Close gripper
                action = np.concatenate([np.zeros(3), np.array([-1.0])])
            else:
                # Open gripper
                action = np.concatenate([np.zeros(3), np.array([1.0])])

            for _ in range(10):
                self.ground_obs_next, _, _, _, _ = self.ground_env.step(action)

                if self.render_mode == "human":
                    ee_pos = self.ground_obs_next[:3]
                    # self.ground_env._env.task.update_waypoints(ee_pos=ee_pos)
                    self.render()
                    time.sleep(self.frame_delay)

            # self.ground_env._env.task.reset_blue_block_pose()

    def _handle_gripper_state_change(self):
        # FIXME: Most of this logic should (probably) be moved to the ground environment
        if self.action[-1] <= 0:
            # Close gripper
            action = np.concatenate([np.zeros(3), np.array([-1.0])])
        else:
            # Open gripper
            action = np.concatenate([np.zeros(3), np.array([1.0])])

        for _ in range(5):
            self.ground_obs_next, _, _, _, _ = self.ground_env.step(action)

            if self.render_mode == "human":
                ee_pos = self.ground_obs_next[:3]
                # self.ground_env._env.task.update_waypoints(ee_pos=ee_pos)
                self.render()
                time.sleep(self.frame_delay)

    def _get_subtask_logs(self) -> dict:
        return {
            "subtask": {
                "green_one_complete": self.green_one_complete,
                "green_two_complete": self.green_two_complete,
                "green_three_complete": self.green_three_complete,
                "red_one_complete": self.red_one_complete,
                "red_two_complete": self.red_two_complete,
                "red_three_complete": self.red_three_complete,
                "blue_one_complete": self.blue_one_complete,
                "blue_two_complete": self.blue_two_complete,
                "blue_three_complete": self.blue_three_complete,
            },
            "stage": {
                "green_above_3_0": self.green_above_3_0,
                "green_above_2_0": self.green_above_2_0,
                "green_above_1_0": self.green_above_1_0,
                "green_grasp_3_0": self.green_grasp_3_0,
                "green_grasp_2_0": self.green_grasp_2_0,
                "green_grasp_1_0": self.green_grasp_1_0,
                "green_release_3_0": self.green_release_3_0,
                "green_release_2_0": self.green_release_2_0,
                "green_release_1_0": self.green_release_1_0,
                "red_above_3_0": self.red_above_3_0,
                "red_above_2_0": self.red_above_2_0,
                "red_above_1_0": self.red_above_1_0,
                "red_grasp_3_0": self.red_grasp_3_0,
                "red_grasp_2_0": self.red_grasp_2_0,
                "red_grasp_1_0": self.red_grasp_1_0,
                "red_release_3_0": self.red_release_3_0,
                "red_release_2_0": self.red_release_2_0,
                "red_release_1_0": self.red_release_1_0,
                "blue_above_3_0": self.blue_above_3_0,
                "blue_above_2_0": self.blue_above_2_0,
                "blue_above_1_0": self.blue_above_1_0,
                "blue_grasp_3_0": self.blue_grasp_3_0,
                "blue_grasp_2_0": self.blue_grasp_2_0,
                "blue_grasp_1_0": self.blue_grasp_1_0,
                "blue_release_3_0": self.blue_release_3_0,
                "blue_release_2_0": self.blue_release_2_0,
                "blue_release_1_0": self.blue_release_1_0,
            },
            "is_success": 1 if self.u in self.crm.F else 0,
        }

    def _init_subtask_logging(self):
        # Green block logging
        self.green_one_complete = False
        self.green_two_complete = False
        self.green_three_complete = False

        self.green_above_3_0 = False
        self.green_above_2_0 = False
        self.green_above_1_0 = False

        self.green_grasp_3_0 = False
        self.green_grasp_2_0 = False
        self.green_grasp_1_0 = False

        self.green_release_3_0 = False
        self.green_release_2_0 = False
        self.green_release_1_0 = False

        # Red block logging
        self.red_one_complete = False
        self.red_two_complete = False
        self.red_three_complete = False

        self.red_above_3_0 = False
        self.red_above_2_0 = False
        self.red_above_1_0 = False

        self.red_grasp_3_0 = False
        self.red_grasp_2_0 = False
        self.red_grasp_1_0 = False

        self.red_release_3_0 = False
        self.red_release_2_0 = False
        self.red_release_1_0 = False

        # Blue block logging
        self.blue_one_complete = False
        self.blue_two_complete = False
        self.blue_three_complete = False

        self.blue_above_3_0 = False
        self.blue_above_2_0 = False
        self.blue_above_1_0 = False

        self.blue_grasp_3_0 = False
        self.blue_grasp_2_0 = False
        self.blue_grasp_1_0 = False

        self.blue_release_3_0 = False
        self.blue_release_2_0 = False
        self.blue_release_1_0 = False


if __name__ == "__main__":
    # solve returns = 196
    import matplotlib.pyplot as plt

    import experiments.warehouse.lib.groundenv.constants.waypoints as cw

    env = ContextSensitiveCrossProductMDP(render_mode="human")
    obs, _ = env.reset()
    returns = 0
    rewards = []

    for i in range(5000):
        ee_pos = obs[:3]

        gripper_action = np.array([-1.0])

        if env.u == 0:
            delta = np.array([0.0, 0.0, 0.1])
        elif env.u == 15:
            delta = cw.ABOVE_RED - ee_pos
        elif env.u == 14:
            delta = cw.GRASP_RED - ee_pos
        elif env.u == 13:
            delta = cw.GRASP_RED - ee_pos
        elif env.u == 12:
            delta = cw.RELEASE_RED - ee_pos
        elif env.u == 11:
            delta = cw.RELEASE_RED - ee_pos
        elif env.u == 10:
            delta = cw.ABOVE_GREEN - ee_pos
        elif env.u == 9:
            delta = cw.GRASP_GREEN - ee_pos
        elif env.u == 8:
            delta = cw.GRASP_GREEN - ee_pos
        elif env.u == 7:
            delta = cw.RELEASE_GREEN - ee_pos
        elif env.u == 6:
            delta = cw.RELEASE_GREEN - ee_pos
        elif env.u == 5:
            delta = cw.ABOVE_BLUE - ee_pos
        elif env.u == 4:
            delta = cw.GRASP_BLUE - ee_pos
        elif env.u == 3:
            delta = cw.GRASP_BLUE - ee_pos
        elif env.u == 2:
            delta = cw.RELEASE_BLUE - ee_pos
        elif env.u == 1:
            delta = cw.RELEASE_BLUE - ee_pos
        else:
            raise ValueError(f"Invalid state {env._u}")

        if env.u in (0, 5, 4, 1, 10, 9, 6, 15, 14, 11):
            # Open gripper
            gripper_action = np.array([1.0])
        else:
            # Close gripper
            gripper_action = np.array([-1.0])

        delta *= 2
        delta = np.clip(delta, -0.25, 0.25)

        action = np.concatenate([delta, gripper_action])

        obs, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        rewards.append(reward)

        env.render()
        time.sleep(0.03)
        if terminated or truncated:
            print(f"Terminated at step {i}")
            break

    rewards = np.array(rewards)
    plt.plot(rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    # plt.ylim(-3, 3)
    plt.title("Reward for a sample trajectory within an episode")
    plt.show()
