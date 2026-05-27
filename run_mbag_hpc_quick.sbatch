#!/bin/bash
# Keep the quick test identifiable in `squeue` and in the scheduler history.
#SBATCH --job-name=mbag-hpc-quick
# Replace this with your actual Slurm account; PIK requires an explicit account.
#SBATCH --account=YOUR_ACCOUNT
# Use the GPU partition because even the smoke test checks CUDA/container access.
#SBATCH --partition=gpu
# `gpushort` is the cheapest GPU QOS for a short validation run.
#SBATCH --qos=gpushort
# Reserve one GPU so Apptainer can expose a real NVIDIA device via `--nv`.
#SBATCH --gres=gpu:1
# Keep the job on a single node; this script does not use multi-node parallelism.
#SBATCH --nodes=1
# One task means one shell process driving one Python suite command.
#SBATCH --ntasks=1
# Give Ray/RLlib a moderate CPU budget without over-requesting for a smoke run.
#SBATCH --cpus-per-task=16
# Reserve enough RAM for the container, Ray object store, and temporary rollout data.
#SBATCH --mem=64G
# Two hours leaves headroom for queue jitter and startup while still enforcing a cap.
#SBATCH --time=02:00:00
# `%x` expands to the job name and `%j` to the numeric job id for easy log lookup.
#SBATCH --output=%x-%j.out
# Keep stderr separate from stdout so launch/config failures are easy to spot.
#SBATCH --error=%x-%j.err

# Fail on command errors, unset variables, and broken pipelines.
set -euo pipefail

# Default to the submission directory so `sbatch` from the repo root just works.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Allow overriding the image path, but prefer a repo-local Apptainer image by default.
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/mbag-hpc.sif}"
# Use the shipped BC checkpoint unless the caller points at another human model.
HUMAN_CHECKPOINT="${HUMAN_CHECKPOINT:-$REPO_DIR/data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100}"
# Keep quick-test outputs separate from full-suite outputs.
OUT_DIR="${OUT_DIR:-$REPO_DIR/data/quickstart_suite_hpc_quick}"
# Put large transient files on the parallel scratch filesystem recommended by the cluster docs.
TMP_ROOT="${TMP_ROOT:-/p/tmp/$USER/mbag/${SLURM_JOB_ID}}"

mkdir -p "$TMP_ROOT"
# Cache Apptainer layers away from `$HOME` to avoid slow home-filesystem traffic.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/p/tmp/$USER/apptainer-cache}"
mkdir -p "$APPTAINER_CACHEDIR"
# Direct generic temp files and Ray scratch space into the job-local scratch area.
export TMPDIR="$TMP_ROOT"
export RAY_TMPDIR="$TMP_ROOT/ray"

cleanup() {
    # Remove scratch files after the job exits so repeated tests do not leak space.
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
    echo "Set OUT_DIR to a fresh path before resubmitting." >&2
    exit 1
fi

# Quick sanity check that the Python stack needed by Ray is present before starting the suite.
apptainer exec --nv "$IMAGE_PATH" python -c "import pkg_resources, ray, torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count(), 'ray', ray.__version__)"

# Run the paper-style suite in its smallest end-to-end configuration.
apptainer exec --nv "$IMAGE_PATH" python -m mbag.scripts.run_paper_experiment_suite \
  --human-checkpoint "$HUMAN_CHECKPOINT" \
  --human-run BC \
  --out-dir "$OUT_DIR" \
    --variant-names standard_paper \
    --train-config-updates-json '{"sample_batch_size":64,"train_batch_size":32,"replay_buffer_size":64}' \
  --local-cpu \
  --quick