import pytest

pytest.importorskip("jax.numpy")
pytest.importorskip("matplotlib")

from examples.jax.letter_q_learning_speed import run_benchmark, save_plots


def test_letter_q_learning_speed_demo_smoke(tmp_path) -> None:
    """The Letter World JAX benchmark runs with tiny settings."""
    result = run_benchmark(episodes=2, max_steps=4, repeats=1, seed=0, counter_cap=2)
    plot_paths = save_plots(result, tmp_path, smoothing_window=1)

    assert result["config"]["episodes"] == 2
    assert result["python"]["mean_s"] >= 0.0
    assert result["python_counterfactual"]["mean_s"] >= 0.0
    assert result["jax"]["compile_first_call_s"] >= 0.0
    assert result["jax_counterfactual"]["compile_first_call_s"] >= 0.0
    assert result["jax"]["steady_mean_s"] >= 0.0
    assert result["jax_counterfactual"]["steady_mean_s"] >= 0.0
    assert "steady_state_ratio" in result["speedup"]
    assert "counterfactual_steady_state_ratio" in result["speedup"]
    assert "python_returns" in result["history"]
    assert "python_counterfactual_returns" in result["history"]
    assert "jax_returns" in result["history"]
    assert "jax_counterfactual_returns" in result["history"]
    assert (tmp_path / "letter_q_learning_returns.png").exists()
    assert (tmp_path / "letter_q_learning_speedup.png").exists()
    assert set(plot_paths) == {"returns_curve", "speedup"}
