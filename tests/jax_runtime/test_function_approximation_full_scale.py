from pathlib import Path

import pytest

from examples.jax.function_approximation_full_scale import (
    compute_reward_cap,
    has_converged,
    normalize_return,
    run_full_benchmark,
    select_best_candidate,
    write_plots,
)


def test_reward_caps_for_all_function_approximation_tasks():
    """Reward caps match the positive target-hit sequences."""
    assert compute_reward_cap("rm-discrete") == 1020.0
    assert compute_reward_cap("crm-discrete") == 40.0
    assert compute_reward_cap("rm-continuous") == 1020.0
    assert compute_reward_cap("crm-continuous") == 220.0


def test_normalized_score_uses_random_to_oracle_gap():
    """Normalized score is anchored at random=0 and oracle=1."""
    target = {
        "random": {"mean_return": -10.0},
        "oracle": {"mean_return": 90.0},
        "oracle_gap": 100.0,
    }

    assert normalize_return(-10.0, target) == 0.0
    assert normalize_return(90.0, target) == 1.0
    assert normalize_return(40.0, target) == 0.5


def test_convergence_requires_min_budget_and_consecutive_hits():
    """Adaptive convergence waits for the minimum budget and recent target hits."""
    curve = [
        {"timesteps": 10, "normalized_score": 0.95},
        {"timesteps": 20, "normalized_score": 0.96},
        {"timesteps": 30, "normalized_score": 0.97},
    ]

    assert not has_converged(
        curve,
        min_timesteps=40,
        required_consecutive=3,
        target_score=0.9,
    )
    assert has_converged(
        curve,
        min_timesteps=30,
        required_consecutive=3,
        target_score=0.9,
    )
    curve[-1]["normalized_score"] = 0.1
    assert not has_converged(
        curve,
        min_timesteps=30,
        required_consecutive=3,
        target_score=0.9,
    )


def test_select_best_calibration_candidate_prefers_solved_then_score():
    """Calibration ranking prefers solved candidates, then score, then speed."""
    candidates = [
        {
            "solved": False,
            "final_normalized_score": 0.8,
            "train_time_s": 1.0,
            "hyperparams": {"learning_rate": 1e-4},
        },
        {
            "solved": True,
            "final_normalized_score": 0.7,
            "train_time_s": 3.0,
            "hyperparams": {"learning_rate": 3e-4},
        },
        {
            "solved": True,
            "final_normalized_score": 0.9,
            "train_time_s": 5.0,
            "hyperparams": {"learning_rate": 1e-3},
        },
    ]

    assert select_best_candidate(candidates)["hyperparams"]["learning_rate"] == 1e-3


def test_full_scale_plot_generation_with_solved_and_unsolved_runs(tmp_path: Path):
    """Plot bundle handles solved and unsolved final runs."""
    result = {
        "targets": {
            "rm-discrete": {
                "target_score": 0.9,
                "target_return": 100.0,
            }
        },
        "runs": [
            {
                "task": "rm-discrete",
                "backend": "sb3",
                "algorithm": "dqn",
                "solved": True,
                "returns_curve": [
                    {
                        "time_s": 0.1,
                        "mean_return": 90.0,
                        "normalized_score": 0.9,
                    }
                ],
            },
            {
                "task": "rm-discrete",
                "backend": "sbx",
                "algorithm": "dqn",
                "solved": False,
                "returns_curve": [
                    {
                        "time_s": 0.2,
                        "mean_return": 50.0,
                        "normalized_score": 0.5,
                    }
                ],
            },
        ],
        "summary": [
            {
                "task": "rm-discrete",
                "backend": "sb3",
                "algorithm": "dqn",
                "train_time_s": 1.0,
                "solved": True,
            },
            {
                "task": "rm-discrete",
                "backend": "sbx",
                "algorithm": "dqn",
                "train_time_s": 2.0,
                "solved": False,
                "speedup_vs_sb3": 0.5,
            },
        ],
    }

    paths = write_plots(result, tmp_path)

    assert set(paths) == {
        "returns_curve",
        "normalized_returns_curve",
        "times",
        "speedups",
    }
    for path in paths.values():
        assert Path(path).exists()


@pytest.mark.parametrize("backend", ["sb3"])
def test_full_scale_runner_tiny_budget_smoke(tmp_path: Path, backend: str):
    """Run a tiny full-scale pass without launching expensive training."""
    result = run_full_benchmark(
        tasks=["rm-discrete"],
        algorithms=["dqn"],
        backends=[backend],
        seed=321,
        repeats=1,
        target_score=0.9,
        max_budget_multiplier=1.0,
        eval_episodes=1,
        baseline_episodes=1,
        required_consecutive=1,
        buffer_size=128,
        batch_size=32,
        learning_starts=100,
        device="cpu",
        json_output=tmp_path / "benchmark.json",
        plot_output_dir=tmp_path,
        max_steps_override=5,
        min_budget_override=2,
        max_budget_override=2,
        calibration_budget_override=2,
        eval_freq_override=1,
        max_calibration_candidates=1,
        resume=False,
    )

    assert (tmp_path / "benchmark.json").exists()
    assert result["targets"]["rm-discrete"]["reward_cap"] == 1020.0
    assert len(result["runs"]) == 1
    assert result["runs"][0]["phase"] == "final"
