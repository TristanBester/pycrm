import time
from warnings import filterwarnings

import gymnasium as gym

from crm.agents.sb3.sac import CounterfactualSAC

filterwarnings("ignore")

PATH = (
    "/Users/tristan/Projects/counting-reward-machines/experiments/warehouse/core/auto/"
    "downloaded_checkpoints/C-SAC_joints_p-1_4/model_6100000_steps.zip"
)


def create_env():
    """Create the warehouse environment."""
    return gym.make(
        "Warehouse-v0",
        **{
            "ground_env_kwargs": {
                "control_type": "joints",
                "render_mode": "human",
                "show_waypoints": True,
                "show_ee_identifier": True,
            },
            "crm_kwargs": {},
            "lf_kwargs": {},
            "crossproduct_kwargs": {
                "max_steps": 200,
            },
        },
    )


def main() -> None:
    """Main function."""
    env = create_env()
    # Load the latest checkpoint
    # checkpoint_path = get_latest_checkpoint()
    model = CounterfactualSAC.load(PATH)

    # Run 10 episodes
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # FORCE OPEN FOR TESTING!!!!

        print(f"\nStarting Episode {episode + 1}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            env.ground_env.task.scene_manager.update_ee_identifier(obs[:3])  # type: ignore
            print("ACTION", action[-1])
            print("OBS", round(obs[6], 4))

            last_c = env.c  # type: ignore

            # lf = WarehouseLabellingFunction()

            # pos_err = obs[:3] - cw.ABOVE_RED
            # print("POS ERR", pos_err.max())
            # print("P", lf(obs, action, obs))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward  # type: ignore

            if env.c != last_c:  # type: ignore
                env.ground_env.task.scene_manager.animate_red_block()  # type: ignore

            print(f"Reward: {reward}")
            print(env.props)  # type: ignore
            # input()
            steps += 1

            # Optional: render the environment if available
            env.render()
            time.sleep(0.01)

        print(f"Episode {episode + 1} finished:")
        print(f"Total Reward: {total_reward}")
        print(f"Steps: {steps}")

    env.close()


if __name__ == "__main__":
    main()
