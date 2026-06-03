#!/bin/bash
# Keep the 10%-run rollout jobs easy to identify in `squeue` and accounting output.
#SBATCH --job-name=mbag-hpc-rollout-10pct
# Replace this with your real Slurm account; the cluster expects explicit charging.
#SBATCH --account=YOUR_ACCOUNT
# Use the GPU partition because rollout.py will use a GPU when one is available.
#SBATCH --partition=gpu
# `gpushort` is a good fit for two-episode rollout extraction jobs.
#SBATCH --qos=gpushort
# Each rollout task loads one assistant checkpoint and evaluates it on one GPU.
#SBATCH --gres=gpu:1
# Keep each array task on one node; the split across variants and seeds comes from Slurm.
#SBATCH --nodes=1
# One Slurm task launches one rollout process for one `(variant, seed)` pair.
#SBATCH --ntasks=1
# Give evaluation workers enough CPU headroom without over-requesting.
#SBATCH --cpus-per-task=8
# Reserve moderate RAM for loading the trainer, env workers, and result writing.
#SBATCH --mem=32G
# Two rollout episodes should fit comfortably in the short GPU queue.
#SBATCH --time=02:00:00
# Nine tasks cover three variants across three seeds; gpushort only allows two active jobs per user.
#SBATCH --array=0-8%2
# `%A` is the array master job id and `%a` the array index, which keeps logs distinct.
#SBATCH --output=%x-%A_%a.out
# Keep stderr separate for each array element to simplify failure diagnosis.
#SBATCH --error=%x-%A_%a.err

# Fail on command errors, unset variables, and broken pipelines.
set -euo pipefail

# Map array indices to explicit `(variant, seed)` pairs so each task is easy to place.
VARIANTS=(standard_paper cluttered_large_grid cluttered_large_grid_goal_agnostic)
SEEDS=(0 1 2)
NUM_SEEDS="${#SEEDS[@]}"
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

# Default to the submission directory so the script works when submitted from the repo root.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Allow overriding the image path, but prefer a repo-local Apptainer image by default.
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/mbag-hpc.sif}"
# Default to the split 10%-run output root written by the training script.
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/data/paper_suite_hpc_bc_split_10pct}"
TRAIN_ROOT="$OUT_ROOT/$VARIANT/seed_${SEED}/train"
# Two rollout episodes per checkpoint by default.
NUM_EPISODES="${NUM_EPISODES:-2}"
# Keep rollout evaluation lightweight but parallel.
NUM_WORKERS="${NUM_WORKERS:-4}"
# Put large transient files on the parallel scratch filesystem recommended by the cluster docs.
TMP_ROOT="${TMP_ROOT:-/p/tmp/$USER/mbag/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

mkdir -p "$TMP_ROOT"
# Cache Apptainer layers away from `$HOME` to avoid slow home-filesystem traffic.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/p/tmp/$USER/apptainer-cache}"
mkdir -p "$APPTAINER_CACHEDIR"
# Direct generic temp files and Ray scratch space into the job-local scratch area.
export TMPDIR="$TMP_ROOT"
export RAY_TMPDIR="$TMP_ROOT/ray"

cleanup() {
    # Remove scratch files after each array element exits so space is reclaimed promptly.
    rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

# Run relative paths from the repository root.
cd "$REPO_DIR"

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Container image not found: $IMAGE_PATH" >&2
    echo "Build it first with: apptainer build mbag-hpc.sif apptainer/mbag-hpc.def" >&2
    exit 1
fi

if [[ ! -d "$TRAIN_ROOT" ]]; then
    echo "Training directory not found: $TRAIN_ROOT" >&2
    exit 1
fi

LATEST_RUN_DIR="$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail -n 1)"
if [[ -z "$LATEST_RUN_DIR" ]]; then
    echo "No numbered Sacred train run found under: $TRAIN_ROOT" >&2
    exit 1
fi

CHECKPOINT_PATH="$(find "$LATEST_RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' | sort -V | tail -n 1)"
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "No checkpoint_* directory found under: $LATEST_RUN_DIR" >&2
    exit 1
fi

# Quick sanity check that the Python stack needed by Ray is present before starting rollout.
apptainer exec --nv "$IMAGE_PATH" python -c "import pkg_resources, ray, torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count(), 'ray', ray.__version__)"

# Run two rollout episodes from the latest assistant checkpoint for this variant and seed.
apptainer exec --nv "$IMAGE_PATH" python -m mbag.scripts.rollout with \
  run=MbagAlphaZero \
  checkpoint="$CHECKPOINT_PATH" \
  experiment_name="${VARIANT}_seed_${SEED}" \
  num_episodes="$NUM_EPISODES" \
  num_workers="$NUM_WORKERS"