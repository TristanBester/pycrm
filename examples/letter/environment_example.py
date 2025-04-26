from examples.letter.core.crossproduct import LetterWorldCrossProduct
from examples.letter.core.ground import LetterWorld
from examples.letter.core.label import LetterWorldLabellingFunction
from examples.letter.core.machine import LetterWorldCountingRewardMachine


def main():
    """In this example, we demonstrate the cross-product MDP.

    More specifically, we show that objects of the cross-product behave
    exactly like objects of the gymnasium.Env class. That is, the cross-product
    MDP is exactly like any other MDP, but it has additional functionality
    for counterfactual experience generation.
    """
    # Create the ground environment
    ground_env = LetterWorld()
    print("Created ground environment with:")
    print(f"  Action space: {ground_env.action_space}")
    print(f"  Observation space: {ground_env.observation_space}")

    # Create labelling function and counting reward machine
    lf = LetterWorldLabellingFunction()
    crm = LetterWorldCountingRewardMachine()

    # Create the cross-product environment
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=100,  # Set maximum number of steps
    )

    print("\nCreated cross-product environment with:")
    print(f"  Action space: {cross_product.action_space}")
    print(f"  Observation space: {cross_product.observation_space}")

    # Reset the environment
    obs, info = cross_product.reset(seed=42)
    print("\nInitial state:")
    print(f"  Observation: {obs}")
    print(f"  Info: {info}")

    # Render the initial state
    print("\nInitial environment state:")
    ground_env.render()

    # Run a few steps with random actions
    print("\nRunning 10 random steps:")
    total_reward = 0

    for step in range(10):
        # Sample a random action - just like in any Gym environment
        action = cross_product.action_space.sample()

        # Step the environment forward
        next_obs, reward, terminated, truncated, info = cross_product.step(action)

        # Track total reward
        total_reward += reward

        # Print information about the step
        print(f"\nStep {step+1}:")
        print(f"  Action: {action}")
        print(f"  Observation: {next_obs}")
        print(f"  Reward: {reward}")
        print(f"  Total reward: {total_reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")

        # Render the environment to visualize the current state
        print(f"\nEnvironment state after step {step+1}:")
        ground_env.render()

        # Check if the episode is done
        if terminated or truncated:
            print("\nEpisode finished after {} steps".format(step + 1))
            break

    # You can also manually reset and interact with the environment
    print("\nResetting environment and taking specific actions:")
    obs, info = cross_product.reset(seed=0)

    # Render the initial state after reset
    print("\nInitial environment state after reset:")
    ground_env.render()

    # Define a sequence of actions (0=RIGHT, 1=LEFT, 2=UP, 3=DOWN)
    actions = [0, 0, 0, 2, 1, 1]  # Example sequence

    for i, action in enumerate(actions):
        next_obs, reward, terminated, truncated, info = cross_product.step(action)
        print(f"\nStep {i+1} with action {action}:")
        print(f"  Reward: {reward}")
        print(f"  Observation: {next_obs}")

        # Render the environment after each step
        print(f"\nEnvironment state after step {i+1}:")
        ground_env.render()

        if terminated or truncated:
            print(f"Episode ended after {i+1} steps")
            break

    print(
        "\nAs shown, the cross-product environment follows the standard Gym interface:"
    )
    print("- It has observation_space and action_space attributes")
    print("- It implements reset() and step() methods with standard returns")
    print("- It can be used in any RL algorithm that works with Gym environments")
    print("- It adds counters to track symbols for reward machine calculation")


if __name__ == "__main__":
    main()
