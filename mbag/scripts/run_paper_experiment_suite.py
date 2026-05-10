import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

DEFAULT_WORLD_SIZE = (11, 10, 10)
DEFAULT_ASSISTANT_EVAL_NUM_SIMULATIONS = 20
DEFAULT_GOAL_SUBSET = "test"
EXCLUDED_CLUTTER_VERTICAL_LAYERS = 3


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    description: str
    train_updates: Dict[str, Any]
    eval_env_updates: Dict[str, Any]


def _sacred_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _make_override_args(overrides: Mapping[str, Any]) -> List[str]:
    return [f"{key}={_sacred_value(value)}" for key, value in overrides.items()]


def _compute_large_grid_variant(
    *,
    clutter_density: float,
    clutter_bedrock_fraction: float,
    world_size: Tuple[int, int, int] = DEFAULT_WORLD_SIZE,
    x_scale: int = 3,
) -> Dict[str, Any]:
    width, height, depth = world_size
    large_world_size = (width * x_scale, height, depth)
    # Clutter is only placed above the floor and below the top buffer in MbagEnv,
    # so exclude the floor and top-buffer layers when converting density to a count.
    clutter_volume = (
        large_world_size[0]
        * large_world_size[2]
        * max(height - EXCLUDED_CLUTTER_VERTICAL_LAYERS, 1)
    )
    num_clutter_blocks = int(round(clutter_density * clutter_volume))
    return {
        "width": large_world_size[0],
        "height": large_world_size[1],
        "depth": large_world_size[2],
        "goal_x_slots": x_scale,
        "num_clutter_blocks": num_clutter_blocks,
        "clutter_bedrock_fraction": clutter_bedrock_fraction,
        "world_size": large_world_size,
    }


def get_default_variants(
    *,
    clutter_density: float,
    clutter_bedrock_fraction: float,
) -> List[ExperimentVariant]:
    large_grid_config = _compute_large_grid_variant(
        clutter_density=clutter_density,
        clutter_bedrock_fraction=clutter_bedrock_fraction,
    )
    train_env_updates = {
        "width": large_grid_config["width"],
        "height": large_grid_config["height"],
        "depth": large_grid_config["depth"],
        "goal_x_slots": large_grid_config["goal_x_slots"],
        "num_clutter_blocks": large_grid_config["num_clutter_blocks"],
        "clutter_bedrock_fraction": large_grid_config["clutter_bedrock_fraction"],
    }
    eval_env_updates = {
        "world_size": large_grid_config["world_size"],
        "goal_x_slots": large_grid_config["goal_x_slots"],
        "num_clutter_blocks": large_grid_config["num_clutter_blocks"],
        "clutter_bedrock_fraction": large_grid_config["clutter_bedrock_fraction"],
    }
    return [
        ExperimentVariant(
            name="standard_paper",
            description="Paper baseline AssistanceZero assistant setting.",
            train_updates={},
            eval_env_updates={},
        ),
        ExperimentVariant(
            name="cluttered_large_grid",
            description="Paper baseline with a 3x wider cluttered grid.",
            train_updates=train_env_updates,
            eval_env_updates=eval_env_updates,
        ),
        ExperimentVariant(
            name="cluttered_large_grid_goal_agnostic",
            description=(
                "Cluttered large-grid variant with goal_agnostic enabled for the "
                "assistant."
            ),
            train_updates={**train_env_updates, "goal_agnostic": True},
            eval_env_updates=eval_env_updates,
        ),
    ]


def build_train_command(
    *,
    python_executable: str,
    human_checkpoint: str,
    human_checkpoint_name: str,
    experiment_dir: Path,
    seed: int,
    variant: ExperimentVariant,
) -> List[str]:
    overrides: Dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "checkpoint_to_load_policies": human_checkpoint,
        "checkpoint_name": human_checkpoint_name,
        "seed": seed,
    }
    overrides.update(variant.train_updates)
    return [
        python_executable,
        "-m",
        "mbag.scripts.train",
        "with",
        "assistancezero_assistant",
        *_make_override_args(overrides),
    ]


def build_evaluate_command(
    *,
    python_executable: str,
    human_run: str,
    human_checkpoint: str,
    human_policy_id: str,
    human_algorithm_config_updates: Mapping[str, Any],
    assistant_checkpoint: str,
    out_dir: Path,
    seed: int,
    num_episodes: int,
    num_workers: int,
    assistant_num_simulations: int,
    goal_subset: str,
    variant: ExperimentVariant,
) -> List[str]:
    env_config_updates: Dict[str, Any] = {
        "horizon": 1500,
        "random_start_locations": True,
        "randomize_first_episode_length": False,
        "terminate_on_goal_completion": True,
        "truncate_on_no_progress_timesteps": None,
        "goal_generator_config": {"goal_generator_config": {"subset": goal_subset}},
    }
    env_config_updates.update(variant.eval_env_updates)
    algorithm_config_updates = [
        dict(human_algorithm_config_updates),
        {
            "mcts_config": {
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
                "num_simulations": assistant_num_simulations,
            }
        },
    ]
    overrides = {
        "runs": [human_run, "MbagAlphaZero"],
        "checkpoints": [human_checkpoint, assistant_checkpoint],
        "policy_ids": [human_policy_id, "assistant"],
        "temperatures": [1.0, 1.0],
        "num_episodes": num_episodes,
        "num_workers": num_workers,
        "seed": seed,
        "out_dir": str(out_dir),
        "algorithm_config_updates": algorithm_config_updates,
        "env_config_updates": env_config_updates,
    }
    return [
        python_executable,
        "-m",
        "mbag.scripts.evaluate",
        *_make_override_args(overrides),
    ]


def _run_command(command: Sequence[str], *, dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _latest_numeric_run_dir(experiment_dir: Path) -> Path:
    run_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No Sacred run directories found in {experiment_dir}")
    return max(run_dirs, key=lambda path: int(path.name))


def _find_final_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.rglob("checkpoint_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    return checkpoints[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def extract_comparable_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    mean_metrics = metrics["mean_metrics"]
    episode_metrics = metrics["episode_metrics"]
    if not episode_metrics:
        raise ValueError("metrics.json contained no episode_metrics to summarize")
    assistant_player_metrics = mean_metrics["player_metrics"][-1]
    goal_completion_rate = sum(
        episode_metric["goal_percentage"] == 1.0 for episode_metric in episode_metrics
    ) / len(episode_metrics)
    return {
        "goal_percentage": float(mean_metrics["goal_percentage"]),
        "goal_completion_rate": float(goal_completion_rate),
        "reward": float(mean_metrics["reward"]),
        "assistant_own_reward": float(assistant_player_metrics["own_reward"]),
        "assistant_goal_dependent_reward": float(
            assistant_player_metrics["goal_dependent_reward"]
        ),
        "assistant_goal_independent_reward": float(
            assistant_player_metrics["goal_independent_reward"]
        ),
        "assistant_place_block_accuracy": float(
            assistant_player_metrics["place_block_accuracy"]
        ),
        "assistant_break_block_accuracy": float(
            assistant_player_metrics["break_block_accuracy"]
        ),
    }


def _aggregate_comparable_metrics(
    comparable_metrics_per_seed: Iterable[Mapping[str, float]],
) -> Dict[str, Dict[str, Any]]:
    comparable_metrics_list = list(comparable_metrics_per_seed)
    if not comparable_metrics_list:
        return {}
    keys = list(comparable_metrics_list[0].keys())
    aggregate: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [metrics[key] for metrics in comparable_metrics_list]
        aggregate[key] = {
            "mean": float(statistics.fmean(values)),
            "stdev": float(statistics.stdev(values)) if len(values) > 1 else None,
        }
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a seeded AssistanceZero experiment suite covering the paper baseline, "
            "a cluttered larger-grid variant, and the same variant with "
            "goal_agnostic enabled."
        )
    )
    parser.add_argument("--human-checkpoint", required=True)
    parser.add_argument("--human-run", default="BC")
    parser.add_argument("--human-policy-id", default="human")
    parser.add_argument("--human-algorithm-config-updates", default="{}")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--assistant-num-simulations",
        type=int,
        default=DEFAULT_ASSISTANT_EVAL_NUM_SIMULATIONS,
    )
    parser.add_argument("--goal-subset", default=DEFAULT_GOAL_SUBSET)
    parser.add_argument("--clutter-density", type=float, default=0.05)
    parser.add_argument("--clutter-bedrock-fraction", type=float, default=0.5)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not Path(args.human_checkpoint).exists():
        raise FileNotFoundError(args.human_checkpoint)

    human_algorithm_config_updates = json.loads(args.human_algorithm_config_updates)
    variants = get_default_variants(
        clutter_density=args.clutter_density,
        clutter_bedrock_fraction=args.clutter_bedrock_fraction,
    )
    human_checkpoint_name = Path(args.human_checkpoint).name

    summary: Dict[str, Any] = {
        "human_checkpoint": str(Path(args.human_checkpoint).resolve()),
        "human_run": args.human_run,
        "human_policy_id": args.human_policy_id,
        "assistant_num_simulations": args.assistant_num_simulations,
        "goal_subset": args.goal_subset,
        "seeds": args.seeds,
        "variants": [],
    }

    for variant in variants:
        variant_summary: Dict[str, Any] = {
            "variant": asdict(variant),
            "runs": [],
        }
        for seed in args.seeds:
            train_dir = out_dir / variant.name / f"seed_{seed}" / "train"
            eval_dir = out_dir / variant.name / f"seed_{seed}" / "evaluate"
            train_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            train_command = build_train_command(
                python_executable=args.python_executable,
                human_checkpoint=args.human_checkpoint,
                human_checkpoint_name=human_checkpoint_name,
                experiment_dir=train_dir,
                seed=seed,
                variant=variant,
            )
            _run_command(train_command, dry_run=args.dry_run)

            assistant_checkpoint = ""
            train_run_dir = None
            eval_command = None
            metrics_path = eval_dir / "metrics.json"
            comparable_metrics = None
            if not args.dry_run:
                train_run_dir = _latest_numeric_run_dir(train_dir)
                assistant_checkpoint = str(_find_final_checkpoint(train_run_dir))
                eval_command = build_evaluate_command(
                    python_executable=args.python_executable,
                    human_run=args.human_run,
                    human_checkpoint=args.human_checkpoint,
                    human_policy_id=args.human_policy_id,
                    human_algorithm_config_updates=human_algorithm_config_updates,
                    assistant_checkpoint=assistant_checkpoint,
                    out_dir=eval_dir,
                    seed=seed,
                    num_episodes=args.num_episodes,
                    num_workers=args.num_workers,
                    assistant_num_simulations=args.assistant_num_simulations,
                    goal_subset=args.goal_subset,
                    variant=variant,
                )
                _run_command(eval_command, dry_run=False)
                comparable_metrics = extract_comparable_metrics(
                    _load_json(metrics_path)
                )
            variant_summary["runs"].append(
                {
                    "seed": seed,
                    "train_command": train_command,
                    "train_run_dir": (
                        None if train_run_dir is None else str(train_run_dir)
                    ),
                    "assistant_checkpoint": assistant_checkpoint,
                    "evaluate_command": eval_command,
                    "metrics_path": str(metrics_path),
                    "comparable_metrics": comparable_metrics,
                }
            )

        variant_summary["aggregate_comparable_metrics"] = _aggregate_comparable_metrics(
            [
                run["comparable_metrics"]
                for run in variant_summary["runs"]
                if run["comparable_metrics"] is not None
            ]
        )
        summary["variants"].append(variant_summary)

    summary_path = out_dir / "suite_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Wrote suite summary to {summary_path}")


if __name__ == "__main__":
    main()
