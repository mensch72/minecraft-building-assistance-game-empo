#!/bin/bash
# Keep the reduced-cost full suite easy to identify in `squeue` and accounting output.
#SBATCH --job-name=mbag-hpc-full-10pct
# Replace this with your real Slurm account; the cluster expects explicit charging.
#SBATCH --account=YOUR_ACCOUNT
# Use the GPU partition because the assistant training workload is GPU-backed.
#SBATCH --partition=gpu
# `gpushort` is easier to backfill for per-variant single-seed jobs.
#SBATCH --qos=gpushort
# Each suite element trains one assistant at a time, so request one GPU per task.
#SBATCH --gres=gpu:1
# Keep each array task on one node; the split across variants and seeds comes from Slurm.
#SBATCH --nodes=1
# One Slurm task launches one Python suite process for one `(variant, seed)` pair.
#SBATCH --ntasks=1
# Keep enough CPU headroom for Ray, evaluation workers, and environment steps.
#SBATCH --cpus-per-task=16
# Peak memory is driven by one variant run, not by the total suite length.
#SBATCH --mem=64G
# Ten percent of the training iterations should comfortably fit inside the short GPU queue.
#SBATCH --time=04:00:00
# Nine tasks cover three variants across three seeds; cap concurrency at four GPUs total.
#SBATCH --array=0-8%4
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
# train.py defaults to 500 iterations, so use 50 by default unless the caller overrides it.
SUITE_TRAIN_UPDATES="${SUITE_TRAIN_UPDATES:-{\"num_training_iters\":50}}"

# Default to the submission directory so the script works when submitted from the repo root.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Allow overriding the image path, but prefer a repo-local Apptainer image by default.
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/mbag-hpc.sif}"
# Use the shipped BC checkpoint unless the caller points at another human model.
HUMAN_CHECKPOINT="${HUMAN_CHECKPOINT:-$REPO_DIR/data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100}"
# Store the reduced suite under its own output root so it does not collide with the full run.
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/data/paper_suite_hpc_bc_split_10pct}"
TASK_OUT_DIR="$OUT_ROOT/$VARIANT/seed_${SEED}"
# Put large transient files on the parallel scratch filesystem recommended by the cluster docs.
TMP_ROOT="${TMP_ROOT:-/p/tmp/$USER/mbag/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

mkdir -p "$TMP_ROOT" "$OUT_ROOT"
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

if [[ ! -d "$HUMAN_CHECKPOINT" ]]; then
    echo "Human checkpoint not found: $HUMAN_CHECKPOINT" >&2
    exit 1
fi

if [[ -e "$TASK_OUT_DIR" ]]; then
    echo "Output directory already exists: $TASK_OUT_DIR" >&2
    echo "Set OUT_ROOT to a fresh path before resubmitting." >&2
    exit 1
fi

# Quick sanity check that the Python stack needed by Ray is present before starting the suite.
apptainer exec --nv "$IMAGE_PATH" python -c "import pkg_resources, ray, torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count(), 'ray', ray.__version__)"

# Run one paper-style variant for exactly one seed in this array element.
apptainer exec --nv "$IMAGE_PATH" python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint "$HUMAN_CHECKPOINT" \
  --human-run BC \
  --out-dir "$OUT_ROOT" \
  --variant-names "$VARIANT" \
  --train-config-updates-json "$SUITE_TRAIN_UPDATES" \
  --seeds "$SEED"