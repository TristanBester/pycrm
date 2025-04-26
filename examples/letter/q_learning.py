import time
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

    # Initialize the Q-table as a defaultdict to store state-action values
    # Each state maps to an array of Q-values for each possible action
    q_table = defaultdict(
        lambda: np.zeros(cross_product.action_space.n)  # type: ignore
    )

    # Q-learning hyperparameters
    EPISODES = 5000  # Total training episodes
    LEARNING_RATE = 0.1  # Alpha: how quickly we update Q-values with new information
    DISCOUNT_FACTOR = 0.99  # Gamma: importance of future rewards vs immediate rewards
    EPSILON = 0.1  # Exploration rate: probability of taking random action

    # Track total rewards for visualization
    all_returns = []

    # Train the agent over multiple episodes with progress bar
    pbar = tqdm(range(EPISODES))
    for episode in pbar:
        obs, _ = cross_product.reset()
        episode_return = 0
        done = False

        # Run a single episode
        while not done:
            # Epsilon-greedy action selection:
            # - With probability EPSILON: explore (random action)
            # - Otherwise: exploit (best known action)
            if np.random.random() < EPSILON or np.all(q_table[tuple(obs)] == 0):
                action = np.random.randint(cross_product.action_space.n)  # type: ignore
            else:
                action = int(np.argmax(q_table[tuple(obs)]))

            # Take action and observe result
            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            episode_return += reward
            done = terminated or truncated

            # Q-learning update
            if done:
                td_target = reward  # No future rewards if episode is done
            else:
                td_target = reward + DISCOUNT_FACTOR * np.max(q_table[tuple(next_obs)])

            td_error = td_target - q_table[tuple(obs)][action]
            q_table[tuple(obs)][action] += LEARNING_RATE * td_error

            # Move to next state
            obs = next_obs

        # Record episode return
        all_returns.append(episode_return)

        # Update progress bar with recent average return
        if episode % 10 == 0:
            pbar.set_description(
                f"Episode {episode} | Ave Return: {np.mean(all_returns[-10:]):.2f}"
            )

    # Visualize training progress
    # Plot shows smoothed returns over time to demonstrate learning progress
    plt.figure(figsize=(10, 6))
    all_returns = np.array(all_returns)

    # Apply moving average for smoothing (window size = 50)
    smoothed_returns = np.convolve(all_returns, np.ones((50,)) / 50, mode="valid")

    plt.plot(smoothed_returns)
    plt.title("Q-Learning Performance in Letter World Environment")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (50-episode moving average)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("letter_world_q_learning_performance.png")  # Save figure before showing
    plt.show()

    # Demonstrate learned policy
    print("\nDemonstrating learned policy...")

    for ep in range(10):
        print(f"\nEpisode {ep + 1}")
        obs, _ = cross_product.reset()
        done = False
        total_reward = 0
        step_count = 0

        # Render initial state
        cross_product.render()

        while not done:
            # Display current state and Q-values
            print(f"State: {obs}, Q-values: {q_table[tuple(obs)]}")

            # Select action (mostly greedy with small exploration)
            if np.random.random() < 0.1 or np.all(q_table[tuple(obs)] == 0):
                action = np.random.randint(cross_product.action_space.n)  # type: ignore
                print(f"Taking random action: {action}")
            else:
                action = int(np.argmax(q_table[tuple(obs)]))
                print(f"Taking greedy action: {action}")

            # Execute action
            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            total_reward += reward
            step_count += 1

            print(f"Reward: {reward}, Cumulative: {total_reward}")

            done = terminated or truncated
            obs = next_obs

            # Render environment after action
            cross_product.render()

            # Brief pause for visualization
            time.sleep(0.5)

        print(f"Final state: {obs}")
        print(
            f"Episode {ep + 1} complete. Total reward: {total_reward}, "
            "Steps: {step_count}"
        )
