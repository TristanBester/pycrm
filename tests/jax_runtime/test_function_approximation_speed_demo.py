from examples.jax.function_approximation_speed import run_benchmark


def test_function_approximation_demo_smoke_with_sb3_backend():
    """Run a tiny SB3 function-approximation benchmark."""
    results = run_benchmark(
        tasks=["rm-discrete"],
        algorithms=["dqn"],
        backends=["sb3"],
        timesteps=2,
        repeats=1,
        seed=123,
        max_steps=5,
        eval_freq=1,
        eval_episodes=1,
        buffer_size=128,
        batch_size=32,
        learning_starts=100,
        device="cpu",
        original_example_timesteps=False,
    )

    assert len(results["runs"]) == 1
    run = results["runs"][0]
    assert run["task"] == "rm-discrete"
    assert run["algorithm"] == "dqn"
    assert run["backend"] == "sb3"
    assert run["train_time_s"] > 0
    assert run["returns_curve"]
