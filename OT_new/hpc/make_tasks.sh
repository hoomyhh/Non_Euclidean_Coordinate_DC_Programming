#!/bin/bash
# Usage: make_tasks.sh TASK_FILE NUM_SEEDS CONFIG [CONFIG ...]
# Writes one line "CONFIG SEED METHOD" per independent run.

set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: $0 TASK_FILE NUM_SEEDS CONFIG [CONFIG ...]" >&2
    exit 2
fi
task_file="$1"; num_seeds="$2"; shift 2
methods=(uniform bregman_gap lipschitz full_entropy full_euclidean)

: > "$task_file"
for config in "$@"; do
    for ((seed = 0; seed < num_seeds; seed++)); do
        for method in "${methods[@]}"; do
            echo "$config $seed $method" >> "$task_file"
        done
    done
done
echo "Wrote $(wc -l < "$task_file") tasks to $task_file"
