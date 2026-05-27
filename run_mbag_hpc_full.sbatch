#!/bin/bash
# Keep the production suite easy to identify in `squeue` and accounting output.
#SBATCH --job-name=mbag-hpc-full
# Replace this with your real Slurm account; the cluster expects explicit charging.
#SBATCH --account=YOUR_ACCOUNT
# Use the GPU partition because the assistant training workload is GPU-backed.
#SBATCH --partition=gpu
# `gpulong` matches the multi-day wall time this suite can realistically need.
#SBATCH --qos=gpulong
# Ask for four GPUs so RLlib can use the larger training budget without starving.
#SBATCH --gres=gpu:4
# Keep each seed on one node; this script scales across seeds via a job array instead.
#SBATCH --nodes=1
# One Slurm task launches one Python suite process for one seed.
#SBATCH --ntasks=1
# Allocate a larger CPU budget for environment workers, Ray, and data movement.
#SBATCH --cpus-per-task=32
# Reserve substantial RAM because the full suite trains three variants and uses Ray.
#SBATCH --mem=256G
# Fourteen days gives a realistic cap for a full BC-based suite seed on the cluster.
#SBATCH --time=14-00:00:00
# Launch one array element per seed so seeds can queue, run, and fail independently.
#SBATCH --array=0-2
# `%A` is the array master job id and `%a` the array index, which keeps logs distinct.
#SBATCH --output=%x-%A_%a.out
# Keep stderr separate for each array element to simplify failure diagnosis.
#SBATCH --error=%x-%A_%a.err

# Fail on command errors, unset variables, and broken pipelines.
set -euo pipefail

# Map array indices to explicit seed values.
SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
# Default to no extra suite-wide train overrides.
SUITE_TRAIN_UPDATES='{}'

# Support a script-level `--quick` mode that preserves the full workflow but cuts training.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            SUITE_TRAIN_UPDATES='{"num_training_iters":2}'
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--quick]" >&2
            exit 1
            ;;
    esac
done

# Default to the submission directory so the script works when submitted from the repo root.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Allow overriding the image path, but prefer a repo-local Apptainer image by default.
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/mbag-hpc.sif}"
# Use the shipped BC checkpoint unless the caller points at another human model.
HUMAN_CHECKPOINT="${HUMAN_CHECKPOINT:-$REPO_DIR/data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100}"
# Store all seeds under one output root, then split by `seed_<n>` below.
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/data/paper_suite_hpc_bc}"
OUT_DIR="$OUT_ROOT/seed_${SEED}"
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

if [[ -e "$OUT_DIR" ]]; then
    echo "Output directory already exists: $OUT_DIR" >&2
    echo "Set OUT_ROOT to a fresh path before resubmitting." >&2
    exit 1
fi

# Quick sanity check that the Python stack needed by Ray is present before starting the suite.
apptainer exec --nv "$IMAGE_PATH" python -c "import pkg_resources, ray, torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count(), 'ray', ray.__version__)"

# Run the full paper-style suite for exactly one seed in this array element.
apptainer exec --nv "$IMAGE_PATH" python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint "$HUMAN_CHECKPOINT" \
  --human-run BC \
  --out-dir "$OUT_DIR" \
    --train-config-updates-json "$SUITE_TRAIN_UPDATES" \
  --seeds "$SEED"