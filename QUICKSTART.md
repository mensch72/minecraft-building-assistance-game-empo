# QUICKSTART

This guide gives the fastest path to run MBAG locally on CPU and on an HPC cluster with Slurm.

## 1) Local laptop (CPU): install

Use Python 3.8–3.10 (3.10 recommended).

```bash
cd /home/runner/work/minecraft-building-assistance-game-empo/minecraft-building-assistance-game-empo
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .[rllib,dev]
```

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
pytest -m "not uses_malmo and not uses_cuda"
```

If you want a faster test pass:

```bash
pytest -m "not uses_malmo and not uses_rllib and not uses_cuda and not slow"
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

## 4) HPC with Slurm batch (Apptainer/Singularity)

Build the container once:

```bash
cd /home/runner/work/minecraft-building-assistance-game-empo/minecraft-building-assistance-game-empo
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

REPO=/path/to/minecraft-building-assistance-game-empo
cd "$REPO"

# 1) GPU/container smoke test
apptainer exec --nv mbag-hpc.sif python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 2) First real HPC experiment (small AssistanceZero run)
apptainer exec --nv mbag-hpc.sif python -m mbag.scripts.train with assistancezero_assistant \
  checkpoint_to_load_policies=/path/to/human/model/checkpoint \
  checkpoint_name=human_seed0 \
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
- Replace `/path/to/...` values with real paths on your cluster.
- For CPU-only Slurm jobs, drop `--gres=gpu:1` and set `num_gpus=0 num_gpus_per_worker=0`.
