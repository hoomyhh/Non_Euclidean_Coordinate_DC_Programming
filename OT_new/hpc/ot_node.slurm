#!/bin/bash
# Runs every line of $OT_TASK_FILE as an independent single-core process on
# a quarter of a GPU node, then aggregates each configuration.  The code is
# CPU-only; the GPU is requested only because the allocation
# (naiss2026-3-305-gpu) cannot use the cpu partition.  Submit from OT_new/:
#   sbatch hpc/ot_node.slurm

#SBATCH --job-name=ot-certified
#SBATCH --account=naiss2026-3-305-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=hpc_logs/%x_%j.out
#SBATCH --error=hpc_logs/%x_%j.err

set -euo pipefail

export OT_PROJECT="${OT_PROJECT:-/nobackup/proj/disk/ulio_inverse/personal/akarak/ot}"
export OT_FEATURE_CACHE="$OT_PROJECT/data/cifar10_resnet18_imagenet1k_v1_features.npz"
export OT_OUTPUT_ROOT="$OT_PROJECT/outputs"
export OT_TASK_FILE="$OT_PROJECT/tasks.txt"

if [[ ! -f "$OT_FEATURE_CACHE" ]]; then
    echo "Feature cache not found: $OT_FEATURE_CACHE" >&2
    exit 2
fi

# One core per process; numpy must not spawn its own threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb

echo "Tasks: $(wc -l < "$OT_TASK_FILE") on ${SLURM_CPUS_PER_TASK} cores ($(hostname))"
# Keep going if some tasks fail; their logs are in $OT_OUTPUT_ROOT/*/logs.
xargs -P "${SLURM_CPUS_PER_TASK}" -L 1 bash hpc/run_task.sh < "$OT_TASK_FILE" \
    || echo "Some tasks failed; aggregating what finished." >&2

for config in $(cut -d' ' -f1 "$OT_TASK_FILE" | sort -u); do
    python run_cifar10.py --aggregate-only --output-dir "$OT_OUTPUT_ROOT/$config"
done
