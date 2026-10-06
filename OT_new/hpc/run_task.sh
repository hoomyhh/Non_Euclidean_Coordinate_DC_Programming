#!/bin/bash
# Usage: run_task.sh CONFIG SEED METHOD
# Runs one (configuration, seed, method) and writes its files to
# $OT_OUTPUT_ROOT/CONFIG without aggregating.

set -euo pipefail
config="$1"; seed="$2"; method="$3"
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/configs.sh"
flags="$(config_flags "$config")"

out_dir="$OT_OUTPUT_ROOT/$config"
log_dir="$out_dir/logs"
mkdir -p "$log_dir"

# shellcheck disable=SC2086
python "$here/../run_cifar10.py" \
    --feature-cache "$OT_FEATURE_CACHE" \
    --output-dir "$out_dir" \
    --num-runs $((seed + 1)) --run-start "$seed" \
    --methods "$method" --skip-aggregate \
    $flags \
    > "$log_dir/run_${seed}_${method}.log" 2>&1
echo "done $config $seed $method"
