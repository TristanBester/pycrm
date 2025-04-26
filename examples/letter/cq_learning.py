from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from examples.letter.core.crossproduct import LetterWorldCrossProduct
from examples.letter.core.ground import LetterWorld
from examples.letter.core.label import LetterWorldLabellingFunction
from examples.letter.core.machine import LetterWorldCountingRewardMachine

if __name__ == "__main__":
    # Initialize environment components
    ground_env = LetterWorld()  # Base grid world environment
    lf = LetterWorldLabellingFunction()  # Labels events like seeing symbols A, B, C
    crm = LetterWorldCountingRewardMachine()  # Defines rewards based on symbol sequence
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=500,  # Maximum steps per episode before termination
    )

    # Hyperparameters
    EPISODES = 5000  # Total number of training episodes
    LEARNING_RATE = 0.01  # Learning rate (alpha)
    DISCOUNT_FACTOR = 0.99  # Discount factor (gamma)
    EPSILON = 0.1  # Exploration probability

    #########################################
    # Standard Q-Learning Implementation
    #########################################

    # Initialize Q-table for state-action value function
    q_table = defaultdict(lambda: np.zeros(cross_product.action_space.n))  # type: ignore

    # Track episode returns for performance analysis
    all_returns_ql = []

    # Train Q-learning agent
    pbar = tqdm(range(EPISODES))
    for episode in pbar:
        obs, _ = cross_product.reset()
        episode_return = 0
        done = False

        while not done:
            # Epsilon-greedy policy for action selection
            if np.random.random() < EPSILON or np.all(q_table[tuple(obs)] == 0):
                action = np.random.randint(cross_product.action_space.n)  # type: ignore
            else:
                action = int(np.argmax(q_table[tuple(obs)]))

            # Execute action in environment
            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            episode_return += reward
            done = terminated or truncated

            # Q-learning update equation:
            # Q(s,a) = Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
            if done:
                td_target = reward
            else:
                td_target = reward + DISCOUNT_FACTOR * np.max(q_table[tuple(next_obs)])

            td_error = td_target - q_table[tuple(obs)][action]
            q_table[tuple(obs)][action] += LEARNING_RATE * td_error

            # Transition to next state
            obs = next_obs

        # Record episode performance
        all_returns_ql.append(episode_return)

        # Update progress information
        if episode % 10 == 0:
            pbar.set_description(
                f"Episode {episode} | Ave Return: {np.mean(all_returns_ql[-10:]):.2f}"
            )

    #########################################
    # Counterfactual Q-Learning Implementation
    #########################################

    # Initialize Q-table for counterfactual agent
    q_table = defaultdict(lambda: np.zeros(cross_product.action_space.n))  # type: ignore

    # Track episode returns for performance comparison
    all_returns_cql = []

    # Train counterfactual Q-learning agent
    pbar = tqdm(range(EPISODES))
    for episode in pbar:
        obs, _ = cross_product.reset()
        episode_return = 0
        done = False

        while not done:
            # Epsilon-greedy policy for action selection
            if np.random.random() < EPSILON or np.all(q_table[tuple(obs)] == 0):
                action = np.random.randint(cross_product.action_space.n)  # type: ignore
            else:
                action = int(np.argmax(q_table[tuple(obs)]))

            # Execute action in environment
            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            episode_return += reward
            done = terminated or truncated

            # Process counterfactual experiences
            for o, a, o_, r, d, _ in zip(
                *cross_product.generate_counterfactual_experience(
                    cross_product.to_ground_obs(obs),
                    action,
                    cross_product.to_ground_obs(next_obs),
                ),
                strict=True,
            ):
                # Update Q-values for counterfactual experiences
                if not d:
                    q_table[tuple(o)][a] += LEARNING_RATE * (
                        r
                        + DISCOUNT_FACTOR * np.max(q_table[tuple(o_)])
                        - q_table[tuple(o)][a]
                    )
                else:
                    q_table[tuple(o)][a] += LEARNING_RATE * (r - q_table[tuple(o)][a])

            # Standard Q-learning update for actual experience
            if done:
                td_target = reward
            else:
                td_target = reward + DISCOUNT_FACTOR * np.max(q_table[tuple(next_obs)])

            td_error = td_target - q_table[tuple(obs)][action]
            q_table[tuple(obs)][action] += LEARNING_RATE * td_error

            # Transition to next state
            obs = next_obs

        # Record episode performance
        all_returns_cql.append(episode_return)

        # Update progress information
        if episode % 10 == 0:
            pbar.set_description(
                f"Episode {episode} | Ave Return: {np.mean(all_returns_cql[-10:]):.2f}"
            )

    #########################################
    # Performance Visualization
    #########################################

    plt.figure(figsize=(10, 6))
    all_returns_ql = np.array(all_returns_ql)
    all_returns_cql = np.array(all_returns_cql)

    # Apply 100-episode moving average for smoother visualization
    smoothed_returns_ql = np.convolve(
        all_returns_ql, np.ones((100,)) / 100, mode="valid"
    )
    smoothed_returns_cql = np.convolve(
        all_returns_cql, np.ones((100,)) / 100, mode="valid"
    )

    # Plot learning curves
    plt.plot(smoothed_returns_ql, label="Q-Learning")
    plt.plot(smoothed_returns_cql, label="Counterfactual Q-Learning")
    plt.title("Q-Learning Performance in Letter World Environment")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (100-episode moving average)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save visualization
    plt.savefig("letter_world_cq_learning_performance.png")
    plt.show()
