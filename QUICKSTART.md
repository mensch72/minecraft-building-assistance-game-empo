# QUICKSTART

This guide gives the fastest path to run MBAG locally on CPU and on an HPC cluster with Slurm.

## 1) Local laptop (CPU): install

Use Python 3.8-3.10 (3.10 recommended).
Python 3.11+ is not supported by the pinned RLlib/NumPy stack in this repository.

The standard local setup is: use `pyenv` to provide Python 3.10, then build the
project virtual environment from that interpreter.

If `pyenv` is not installed yet, install it with:

```bash
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Then, from the repository root, run:

```bash
pyenv install -s 3.10.16

PYENV_VERSION=3.10.16 python -V

rm -rf .venv
PYENV_VERSION=3.10.16 python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<25" "setuptools<82" "wheel<0.46" "packaging<25"
pip install -e .[rllib,dev]
```

Sanity checks:

```bash
python -V
which python
```

You should see Python 3.10.x, and `which python` should point to the
repository-local `.venv`.

In a fresh terminal later, reactivate the same environment from the repository
root with:

```bash
cd /path/to/minecraft-building-assistance-game-empo
source .venv/bin/activate
python -V
which python
```

If your shell does not already initialize `pyenv`, rerun the `PYENV_ROOT` /
`PATH` / `pyenv init` lines from above before activating `.venv`.

If your existing `.venv` shows import errors from `pip`, `packaging`, `black`, `ray`, or
`sacred`, delete `.venv` and rerun the install block above from scratch.

If `pyenv install` fails on your OS, install the standard Python build prerequisites for
your platform and rerun the same commands.

Download the CraftAssist house dataset (required for training/evaluation):

```bash
cd data
wget https://minecraft-building-assistance-game.s3.us-east-1.amazonaws.com/craftassist.zip
unzip -o craftassist.zip
cd ..
```

## 2) Local laptop (CPU): quick test

Quick import smoke test:

```bash
python -c "import mbag; print('mbag import OK')"
```

Project lint/test checks:

```bash
./lint.sh
pytest -m "not uses_malmo and not uses_cuda and not slow"
```

The broader non-Malmo/non-CUDA suite includes slow RLlib integration tests and can
spend noticeable time waiting on Ray startup with little CPU activity.

If you want an even faster test pass:

```bash
pytest -m "not uses_malmo and not uses_rllib and not uses_cuda and not slow"
```

If you explicitly want the slower RLlib integration coverage too:

```bash
pytest -m "not uses_malmo and not uses_cuda"
```

## 3) Local laptop (CPU): first actual experiment

Run a tiny PPO training job (CPU-only) as a first real experiment:

```bash
python -m mbag.scripts.train with ppo_human \
  num_training_iters=1 \
  num_workers=0 \
  num_envs_per_worker=1 \
  train_batch_size=256 \
  rollout_fragment_length=64 \
  sgd_minibatch_size=64 \
  num_gpus=0 \
  num_gpus_per_worker=0 \
  save_freq=1
```

Outputs/checkpoints are written under `data/logs/...`. The command logs the final checkpoint path as `final_checkpoint`.

### Local laptop: paper-style assistant suite

If you want the full paper-style comparison on a local machine instead of a single
training run, use the suite launcher directly from the local Python environment. It
runs three experiment types for every seed you provide and writes a
`suite_summary.json` file with per-seed and aggregate results.

Under the hood, these three modes are still regular `python -m mbag.scripts.train`
calls with different overrides. So if you want to run just one of the three modes
manually, you can do that directly with `mbag.scripts.train`.

The three experiment types are:

- Standard paper setting: the original AssistanceZero setup from the paper, with the standard grid and no added clutter.
- Clutter plus random placement: a 3x wider grid with clustered clutter blocks added, where houses are sampled and randomly placed across that wider horizontal build area.
- Clutter plus random placement plus `goal_agnostic=True`: the same cluttered wider-grid setup, but with the assistant trained in goal-agnostic mode.

The shipped BC human checkpoint was trained on the standard `11x10x10` world.
When you switch to the wider `33x10x10` variants, MBAG now reuses the compatible
human-model weights and keeps width-dependent tensors, such as position
embeddings, at the new world size.

Use the shipped BC human checkpoint below for the commands in this section:

```bash
HUMAN_CHECKPOINT=data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100
CHECKPOINT_NAME=sample_human_bc
LOCAL_CPU_TRAIN_OVERRIDES='num_training_iters=1 num_workers=0 num_envs_per_worker=1 sample_batch_size=256 train_batch_size=64 sgd_minibatch_size=128 num_simulations=1 replay_buffer_size=256 simple_optimizer=True num_gpus=0 num_gpus_per_worker=0'
```

`sgd_minibatch_size` must stay larger than the assistant config's `max_seq_len=64`,
so keep it at `128` or another value above `64` in these local CPU examples.
`replay_buffer_size=256` keeps the AlphaZero replay buffer small enough for a
laptop-sized run, and `num_simulations=1` keeps the MCTS budget at smoke-test scale.
`simple_optimizer=True` keeps these local recurrent runs on RLlib's plain CPU
minibatch path instead of the loaded-batch multi-GPU optimizer path.

Direct training commands for those three modes are:

```bash
# 1) Standard paper setting
python -m mbag.scripts.train with assistancezero_assistant \
  $LOCAL_CPU_TRAIN_OVERRIDES \
  checkpoint_to_load_policies="$HUMAN_CHECKPOINT" \
  checkpoint_name="$CHECKPOINT_NAME"

# 2) Clutter + random placement in a wider ambient grid
python -m mbag.scripts.train with assistancezero_assistant \
  $LOCAL_CPU_TRAIN_OVERRIDES \
  checkpoint_to_load_policies="$HUMAN_CHECKPOINT" \
  checkpoint_name="$CHECKPOINT_NAME" \
  width=33 height=10 depth=10 \
  num_clutter_blocks=116 \
  clutter_bedrock_fraction=0.5

# 3) Same as above, but with goal_agnostic=True
python -m mbag.scripts.train with assistancezero_assistant \
  $LOCAL_CPU_TRAIN_OVERRIDES \
  checkpoint_to_load_policies="$HUMAN_CHECKPOINT" \
  checkpoint_name="$CHECKPOINT_NAME" \
  width=33 height=10 depth=10 \
  num_clutter_blocks=116 \
  clutter_bedrock_fraction=0.5 \
  goal_agnostic=True
```

Example:

```bash
rm -rf data/quickstart_suite

python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint "$HUMAN_CHECKPOINT" \
  --human-run BC \
  --out-dir data/quickstart_suite \
  --local-cpu \
  --seeds 0
```

Here `--human-checkpoint` is the input human model. The line above passes the
checkpoint directory itself. You can also pass the Sacred run directory or the
experiment directory that contains numbered runs.

For a fast workflow smoke test, use `--quick`:

```bash
rm -rf data/quickstart_suite_quick

python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint "$HUMAN_CHECKPOINT" \
  --human-run BC \
  --out-dir data/quickstart_suite_quick \
  --quick
```

`--quick` still runs the full suite structure: it keeps the train and evaluate
steps for all three variants, but shrinks the work aggressively by forcing one
seed, zero assistant training iterations, no parallel evaluation workers,
`assistant_num_simulations=1`, and `horizon=16`. It is intended to finish in
well under a minute on a typical local machine.

If you want a small but nonzero local training run instead of the zero-iteration
smoke setting, omit `--quick` and use `--local-cpu` instead.

Useful options:

- `--dry-run` prints the exact train/evaluate commands without executing them.
- `--clutter-density` controls how much clutter is injected into the wider grid variants.
- `--assistant-num-simulations` controls the assistant's evaluation-time MCTS budget.
- `--human-algorithm-config-updates` lets you pass JSON overrides for the human policy during evaluation.

For the two wider-grid variants, the world size becomes `33x10x10`. The clutter is
added to that larger ambient grid, and the goal house is randomly placed within the
wider horizontal build area rather than staying in the original narrower layout.

On a laptop, start with a single seed and a small evaluation budget, because these
suite runs are much slower than the tiny PPO smoke test above.

## 4) HPC with Slurm batch (Apptainer/Singularity)

Build the container once:

```bash
apptainer build mbag-hpc.sif apptainer/mbag-hpc.def
```

Create `run_mbag_quickstart.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=mbag-quickstart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=mbag-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

HUMAN_CHECKPOINT=data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100

# 1) GPU/container smoke test
apptainer exec --nv mbag-hpc.sif python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 2) First real HPC experiment (small AssistanceZero run)
apptainer exec --nv mbag-hpc.sif python -m mbag.scripts.train with assistancezero_assistant \
  checkpoint_to_load_policies="$HUMAN_CHECKPOINT" \
  checkpoint_name=sample_human_bc \
  num_training_iters=2 \
  num_workers=1 \
  num_envs_per_worker=1 \
  sample_batch_size=256 \
  train_batch_size=64 \
  num_gpus=1 \
  num_gpus_per_worker=0
```

Submit with:

```bash
sbatch run_mbag_quickstart.sbatch
```

Notes:

- Keep `--nv` so Apptainer exposes NVIDIA GPUs.
- For CPU-only Slurm jobs, drop `--gres=gpu:1` and set `num_gpus=0 num_gpus_per_worker=0`.

## 5) HPC: paper-style assistant suite

If you want the full paper-style comparison instead of a single training run, use
the suite launcher. It runs three experiment types for every seed you provide and
writes a `suite_summary.json` file with per-seed and aggregate results.

As in the local case, these three modes are just `python -m mbag.scripts.train with
assistancezero_assistant ...` plus different overrides for clutter, wider-grid
placement, and `goal_agnostic=True`.

The three experiment types are:

- Standard paper setting: the original AssistanceZero setup from the paper, with the standard grid and no added clutter.
- Clutter plus random placement: a 3x wider grid with clustered clutter blocks added, where houses are sampled and randomly placed across that wider horizontal build area.
- Clutter plus random placement plus `goal_agnostic=True`: the same cluttered wider-grid setup, but with the assistant trained in goal-agnostic mode.

Example:

```bash
rm -rf data/quickstart_suite_hpc

apptainer exec --nv mbag-hpc.sif python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100 \
  --human-run BC \
  --out-dir data/quickstart_suite_hpc \
  --seeds 0 1 2
```

For a fast HPC smoke test of the orchestration, use `--quick`:

```bash
rm -rf data/quickstart_suite_hpc_quick

apptainer exec --nv mbag-hpc.sif python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100 \
  --human-run BC \
  --out-dir data/quickstart_suite_hpc_quick \
  --quick
```

Here too, `--quick` keeps the full train-plus-evaluate workflow for all three
variants, but forces one seed, zero assistant training iterations, one
evaluation episode, `assistant_num_simulations=1`, and `horizon=16`.

Useful options:

- `--dry-run` prints the exact train/evaluate commands without executing them.
- `--clutter-density` controls how much clutter is injected into the wider grid variants.
- `--assistant-num-simulations` controls the assistant's evaluation-time MCTS budget.
- `--human-algorithm-config-updates` lets you pass JSON overrides for the human policy during evaluation.

For the two wider-grid variants, the world size becomes `33x10x10`. The clutter is
added to that larger ambient grid, and the goal house is randomly placed within the
wider horizontal build area rather than staying in the original narrower layout.
