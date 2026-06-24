import pytest

pytest.importorskip("jax.numpy")
pytest.importorskip("matplotlib")

from examples.jax.tabular_q_learning_speed import run_suite_benchmark, save_plots


def test_tabular_q_learning_speed_suite_smoke(tmp_path) -> None:
    """The full tabular JAX benchmark suite runs with tiny settings."""
    result = run_suite_benchmark(
        episodes=2,
        max_steps=4,
        repeats=1,
        seed=0,
        counter_cap=2,
    )
    plot_paths = save_plots(result, tmp_path, smoothing_window=1)

    assert result["config"]["episodes"] == 2
    assert set(result["tasks"]) == {"letter", "office-rm", "office-crm"}

    for task_result in result["tasks"].values():
        assert task_result["python"]["mean_s"] >= 0.0
        assert task_result["python_counterfactual"]["mean_s"] >= 0.0
        assert task_result["jax"]["compile_first_call_s"] >= 0.0
        assert task_result["jax_counterfactual"]["compile_first_call_s"] >= 0.0
        assert task_result["jax"]["steady_mean_s"] >= 0.0
        assert task_result["jax_counterfactual"]["steady_mean_s"] >= 0.0
        assert "q_learning_steady_state_ratio" in task_result["speedup"]
        assert "counterfactual_steady_state_ratio" in task_result["speedup"]
        assert "python_returns" in task_result["history"]
        assert "python_counterfactual_returns" in task_result["history"]
        assert "jax_returns" in task_result["history"]
        assert "jax_counterfactual_returns" in task_result["history"]

    assert (tmp_path / "tabular_returns.png").exists()
    assert (tmp_path / "tabular_times.png").exists()
    assert (tmp_path / "tabular_speedups.png").exists()
    assert set(plot_paths) == {"returns_curve", "times", "speedups"}
