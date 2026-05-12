from mbag.scripts.run_paper_experiment_suite import (
    LOCAL_CPU_TRAIN_UPDATES,
    ExperimentVariant,
    build_evaluate_command,
    build_train_command,
    extract_comparable_metrics,
    get_default_variants,
    resolve_checkpoint_path,
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
    assert "num_clutter_blocks=116" in command


def test_build_train_command_includes_local_cpu_overrides(tmp_path):
    command = build_train_command(
        python_executable="python",
        human_checkpoint="/tmp/human/checkpoint",
        human_checkpoint_name="checkpoint_000100",
        experiment_dir=tmp_path,
        seed=0,
        variant=ExperimentVariant(
            name="standard_paper",
            description="",
            train_updates={},
            eval_env_updates={},
        ),
        train_config_updates=LOCAL_CPU_TRAIN_UPDATES,
    )

    assert "num_training_iters=1" in command
    assert "num_workers=0" in command
    assert "num_envs_per_worker=1" in command
    assert "sample_batch_size=256" in command
    assert "train_batch_size=64" in command
    assert "sgd_minibatch_size=128" in command
    assert "num_simulations=1" in command
    assert "replay_buffer_size=256" in command
    assert "simple_optimizer=True" in command
    assert "num_gpus=0" in command
    assert "num_gpus_per_worker=0" in command


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
                "num_clutter_blocks": 116,
                "clutter_bedrock_fraction": 0.5,
            },
        ),
    )

    assert command[:4] == ["python", "-m", "mbag.scripts.evaluate", "with"]
    assert 'runs=["BC","MbagAlphaZero"]' in command
    assert 'policy_ids=["human","assistant"]' in command
    assert "explore=False" in command
    assert "seed=3" in command
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


def test_resolve_checkpoint_path_accepts_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint_000123"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "algorithm_state.pkl").write_text("state")

    assert resolve_checkpoint_path(str(checkpoint_dir)) == checkpoint_dir


def test_resolve_checkpoint_path_accepts_sacred_run_dir(tmp_path):
    run_dir = tmp_path / "7"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint_000045"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "algorithm_state.pkl").write_text("state")

    assert resolve_checkpoint_path(str(run_dir)) == checkpoint_dir


def test_resolve_checkpoint_path_accepts_sacred_experiment_dir(tmp_path):
    experiment_dir = tmp_path / "human_run"
    experiment_dir.mkdir()
    older_run_dir = experiment_dir / "3"
    older_run_dir.mkdir()
    older_checkpoint_dir = older_run_dir / "checkpoint_000010"
    older_checkpoint_dir.mkdir()
    (older_checkpoint_dir / "algorithm_state.pkl").write_text("state")
    newer_run_dir = experiment_dir / "11"
    newer_run_dir.mkdir()
    newer_checkpoint_dir = newer_run_dir / "checkpoint_000020"
    newer_checkpoint_dir.mkdir()
    (newer_checkpoint_dir / "algorithm_state.pkl").write_text("state")

    assert resolve_checkpoint_path(str(experiment_dir)) == newer_checkpoint_dir


def test_resolve_checkpoint_path_rejects_non_checkpoint_directory(tmp_path):
    invalid_dir = tmp_path / "jobst"
    invalid_dir.mkdir()

    try:
        resolve_checkpoint_path(str(invalid_dir))
    except ValueError as exc:
        assert "does not resolve to a usable human checkpoint" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-checkpoint directory")
