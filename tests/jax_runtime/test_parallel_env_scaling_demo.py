import pytest

pytest.importorskip("jax.numpy")
pytest.importorskip("matplotlib")

from examples.jax.parallel_env_scaling import run_parallel_env_benchmark, save_plots


def test_parallel_env_scaling_demo_smoke(tmp_path) -> None:
    """The parallel-env JAX benchmark runs with tiny settings."""
    result = run_parallel_env_benchmark(
        task_names=["letter"],
        env_counts=[1, 2],
        steps_per_env=3,
        repeats=1,
        seed=0,
    )
    plot_paths = save_plots(result, tmp_path)

    assert result["config"]["env_counts"] == [1, 2]
    assert set(result["tasks"]) == {"letter"}

    task_result = result["tasks"]["letter"]
    assert len(task_result["python"]["mean_s"]) == 2
    assert len(task_result["jax"]["steady_mean_s"]) == 2
    assert len(task_result["jax"]["compile_first_call_s"]) == 2
    assert len(task_result["speedup"]["steady_state_ratio"]) == 2
    assert all(value > 0 for value in task_result["python"]["steps_per_second"])
    assert all(value > 0 for value in task_result["jax"]["steps_per_second"])

    assert (tmp_path / "parallel_env_throughput.png").exists()
    assert (tmp_path / "parallel_env_times.png").exists()
    assert (tmp_path / "parallel_env_speedup.png").exists()
    assert set(plot_paths) == {"throughput", "times", "speedup"}
