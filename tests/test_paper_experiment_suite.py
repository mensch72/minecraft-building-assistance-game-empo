from mbag.scripts.run_paper_experiment_suite import (
    ExperimentVariant,
    build_evaluate_command,
    build_train_command,
    extract_comparable_metrics,
    get_default_variants,
)


def test_default_variants_include_required_suite_members():
    variants = get_default_variants(
        clutter_density=0.05,
        clutter_bedrock_fraction=0.5,
    )

    assert [variant.name for variant in variants] == [
        "standard_paper",
        "cluttered_large_grid",
        "cluttered_large_grid_goal_agnostic",
    ]
    large_grid_variant = variants[1]
    assert large_grid_variant.train_updates["width"] == 33
    assert large_grid_variant.train_updates["goal_x_slots"] == 3
    assert large_grid_variant.train_updates["num_clutter_blocks"] > 0
    assert variants[2].train_updates["goal_agnostic"] is True


def test_build_train_command_carries_seed_and_variant_updates(tmp_path):
    variant = ExperimentVariant(
        name="cluttered_large_grid_goal_agnostic",
        description="",
        train_updates={
            "width": 33,
            "height": 10,
            "depth": 10,
            "goal_x_slots": 3,
            "num_clutter_blocks": 116,
            "clutter_bedrock_fraction": 0.5,
            "goal_agnostic": True,
        },
        eval_env_updates={},
    )

    command = build_train_command(
        python_executable="python",
        human_checkpoint="/tmp/human/checkpoint",
        human_checkpoint_name="checkpoint_000100",
        experiment_dir=tmp_path,
        seed=7,
        variant=variant,
    )

    assert command[:4] == ["python", "-m", "mbag.scripts.train", "with"]
    assert "assistancezero_assistant" in command
    assert "seed=7" in command
    assert "goal_agnostic=True" in command
    assert "goal_x_slots=3" in command
    assert "num_clutter_blocks=116" in command


def test_build_evaluate_command_uses_comparable_eval_metrics_setup(tmp_path):
    command = build_evaluate_command(
        python_executable="python",
        human_run="BC",
        human_checkpoint="/tmp/human/checkpoint",
        human_policy_id="human",
        human_algorithm_config_updates={},
        assistant_checkpoint="/tmp/assistant/checkpoint",
        out_dir=tmp_path,
        seed=3,
        num_episodes=25,
        num_workers=4,
        assistant_num_simulations=20,
        goal_subset="test",
        variant=ExperimentVariant(
            name="cluttered_large_grid",
            description="",
            train_updates={},
            eval_env_updates={
                "world_size": (33, 10, 10),
                "goal_x_slots": 3,
                "num_clutter_blocks": 116,
                "clutter_bedrock_fraction": 0.5,
            },
        ),
    )

    assert command[:4] == ["python", "-m", "mbag.scripts.evaluate", "with"]
    assert 'runs=["BC","MbagAlphaZero"]' in command
    assert 'policy_ids=["human","assistant"]' in command
    assert "seed=3" in command
    assert any("goal_x_slots" in arg for arg in command)
    assert any('"num_simulations":20' in arg for arg in command)


def test_extract_comparable_metrics_prefers_normalized_outputs():
    metrics = {
        "mean_metrics": {
            "goal_percentage": 0.75,
            "reward": 12.5,
            "player_metrics": [
                {},
                {
                    "own_reward": 5.0,
                    "goal_dependent_reward": 2.0,
                    "goal_independent_reward": 3.0,
                    "place_block_accuracy": 0.9,
                    "break_block_accuracy": 0.8,
                },
            ],
        },
        "episode_metrics": [
            {"goal_percentage": 1.0},
            {"goal_percentage": 0.5},
            {"goal_percentage": 1.0},
            {"goal_percentage": 0.5},
        ],
    }

    comparable_metrics = extract_comparable_metrics(metrics)

    assert comparable_metrics == {
        "goal_percentage": 0.75,
        "goal_completion_rate": 0.5,
        "reward": 12.5,
        "assistant_own_reward": 5.0,
        "assistant_goal_dependent_reward": 2.0,
        "assistant_goal_independent_reward": 3.0,
        "assistant_place_block_accuracy": 0.9,
        "assistant_break_block_accuracy": 0.8,
    }
