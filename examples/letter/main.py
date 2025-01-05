import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from examples.letter.crossproduct import LetterWorldCrossProduct
from examples.letter.ground import LetterWorld
from examples.letter.label import LetterWorldLabellingFunction
from examples.letter.machine import LetterWorldCountingRewardMachine

if __name__ == "__main__":
    ground_env = LetterWorld()
    lf = LetterWorldLabellingFunction()
    crm = LetterWorldCountingRewardMachine()
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=100,
    )

    obs, _ = cross_product.reset()
    cross_product.render()

    for _ in range(100):
        print(obs)
        action = int(input("Action: "))

        obs, reward, terminated, truncated, _ = cross_product.step(action)
        cross_product.render()

    # q_table = defaultdict(lambda: np.zeros(cross_product.action_space.n))  # type: ignore

    # all_returns = []

    # for _ in tqdm(range(10000)):
    #     obs, _ = cross_product.reset()
    #     terminated = False
    #     truncated = False

    #     returns = 0
    #     done = False
    #     while not done:
    #         if np.random.random() < 0.1 or np.all(q_table[tuple(obs)] == 0):
    #             action = np.random.randint(0, 4)
    #         else:
    #             action = int(np.argmax(q_table[tuple(obs)]))
    #         next_obs, reward, terminated, truncated, _ = cross_product.step(action)
    #         returns += reward
    #         done = terminated or truncated
    #         print(obs)

    #         if not done:
    #             q_table[tuple(obs)][action] += 0.1 * (
    #                 reward
    #                 + 0.99 * np.max(q_table[tuple(next_obs)])
    #                 - q_table[tuple(obs)][action]
    #             )
    #         else:
    #             q_table[tuple(obs)][action] += 0.1 * (
    #                 reward - q_table[tuple(obs)][action]
    #             )

    #         obs = next_obs

    #     all_returns.append(returns)

    # all_returns = np.array(all_returns)
    # # Smooth using a rolling average
    # smoothed_returns = np.convolve(all_returns, np.ones((50,)) / 50, mode="valid")
    # plt.plot(smoothed_returns)
    # plt.show()
