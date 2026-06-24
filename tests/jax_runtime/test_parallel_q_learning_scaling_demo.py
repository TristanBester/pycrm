import pytest

pytest.importorskip("jax.numpy")
pytest.importorskip("matplotlib")

from examples.jax.parallel_q_learning_scaling import (
    run_parallel_learning_benchmark,
    save_plots,
)


def test_parallel_q_learning_scaling_demo_smoke(tmp_path) -> None:
    """The parallel-env learning benchmark runs with tiny settings."""
    result = run_parallel_learning_benchmark(
        task_names=["letter"],
        env_counts=[1, 2],
        chunks=2,
        updates_per_chunk=2,
        max_steps=4,
        repeats=1,
        seed=0,
        counter_cap=2,
        eval_episodes=2,
        eval_max_steps=4,
    )
    plot_paths = save_plots(result, tmp_path)

    assert result["config"]["env_counts"] == [1, 2]
    assert set(result["tasks"]) == {"letter"}

    task_result = result["tasks"]["letter"]
    assert set(task_result["env_counts"]) == {"1", "2"}
    for env_result in task_result["env_counts"].values():
        assert env_result["compile_first_call_s"] >= 0.0
        assert len(env_result["mean_wall_time_s"]) == 3
        assert len(env_result["mean_environment_transitions"]) == 3
        assert len(env_result["mean_eval_return"]) == 3
        assert env_result["total_mean_train_time_s"] >= 0.0
        assert env_result["transitions_per_second"] > 0.0

    assert (tmp_path / "parallel_learning_wallclock.png").exists()
    assert (tmp_path / "parallel_learning_samples.png").exists()
    assert (tmp_path / "parallel_learning_time_to_target.png").exists()
    assert set(plot_paths) == {
        "wallclock_returns",
        "sample_returns",
        "time_to_target",
    }
