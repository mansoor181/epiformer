#!/bin/bash
# =============================================================================
# Run m1_noplm_epi_ratio.sh and m1_noplm_epi_group.sh for 3 seeds.
# GPU 0: epitope_ratio seeds sequentially
# GPU 1: epitope_group seeds sequentially
# Both GPUs run in parallel.
#
# Usage:
#   ./scripts/run_m1_noplm_multiseed.sh [--background]
#
# Results: baselines/rebuttals/m1_noplm_multiseed_results.csv
# =============================================================================

if [[ "${1:-}" == "--background" ]]; then
    shift
    SCRIPT_PATH="$(realpath "$0")"
    LOG_FILE="$(dirname "$SCRIPT_PATH")/../logs/m1_noplm_multiseed_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Starting in background. Log: $LOG_FILE"
    nohup "$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M1_PLM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_CSV="/home/exouser/epiformer/baselines/rebuttals/m1_noplm_multiseed_results.csv"

SEEDS=(42 123 456)

echo "============================================"
echo "M1 No-PLM Multi-Seed"
echo "============================================"
echo "Seeds: ${SEEDS[*]}"
echo "epitope_ratio -> GPU 0, epitope_group -> GPU 1"
echo "Results: $RESULTS_CSV"
echo "============================================"

echo "split,seed,auroc,auprc,f1,mcc,precision,recall" > "$RESULTS_CSV"

# -------------------------------------------------------
extract_metrics() {
    local log_file="$1" split="$2" seed="$3"

    if [ ! -f "$log_file" ] || ! grep -q "Final Test Evaluation" "$log_file"; then
        echo "  WARNING: incomplete — $log_file"
        echo "$split,$seed,,,,,," >> "$RESULTS_CSV"
        return
    fi

    local section
    section=$(sed -n '/Final Test Evaluation/,$p' "$log_file")

    local auroc auprc f1 mcc prec rec
    auroc=$(echo "$section" | grep -oP '^epitope_auc: \K[0-9.]+' | tail -1)
    auprc=$(echo "$section" | grep -oP '^epitope_auprc: \K[0-9.]+' | tail -1)
    f1=$(echo "$section" | grep -oP '^epitope_f1: \K[0-9.]+' | tail -1)
    mcc=$(echo "$section" | grep -oP '^epitope_mcc: \K[0-9.-]+' | tail -1)
    prec=$(echo "$section" | grep -oP '^epitope_precision: \K[0-9.]+' | tail -1)
    rec=$(echo "$section" | grep -oP '^epitope_recall: \K[0-9.]+' | tail -1)

    echo "  $split seed=$seed: AUROC=$auroc F1=$f1 MCC=$mcc"
    echo "$split,$seed,$auroc,$auprc,$f1,$mcc,$prec,$rec" >> "$RESULTS_CSV"
}

# -------------------------------------------------------
run_seeds() {
    local split="$1" gpu="$2"
    # Script names use epi_ratio/epi_group, not epitope_ratio/epitope_group
    local short_split="${split/epitope_/epi_}"

    for seed in "${SEEDS[@]}"; do
        echo "[GPU $gpu] $split seed=$seed"

        cd "$M1_PLM_DIR"

        bash "$SCRIPT_DIR/m1_noplm_${short_split}.sh" \
            --gpu_id "$gpu" --batch_size 8 --epochs 130 --server local --seed "$seed" --wandb

        # The script nohup+backgrounds. Find the log it just created and wait.
        local log_file
        log_file=$(ls -t "$M1_PLM_DIR/logs/m1_noplm_${short_split}_seed${seed}_"*_output.log 2>/dev/null | head -1)

        if [ -z "$log_file" ]; then
            echo "  WARNING: no log found for $split seed=$seed"
            echo "$split,$seed,,,,,," >> "$RESULTS_CSV"
            continue
        fi

        echo "  Waiting for completion... (log: $log_file)"
        while true; do
            if grep -q "End-to-End training pipeline completed" "$log_file" 2>/dev/null; then
                break
            fi
            if grep -q "Traceback" "$log_file" 2>/dev/null && \
               ! pgrep -f "run_id=m1_noplm_${short_split}_seed${seed}" > /dev/null 2>&1; then
                echo "  Run appears to have failed."
                break
            fi
            sleep 30
        done

        extract_metrics "$log_file" "$split" "$seed"
    done
}

# -------------------------------------------------------
# Launch both splits in parallel
# -------------------------------------------------------
run_seeds "epitope_ratio" 0 &
pid0=$!

run_seeds "epitope_group" 1 &
pid1=$!

echo "Running in parallel: ratio (GPU 0, PID $pid0), group (GPU 1, PID $pid1)"

wait $pid0
wait $pid1

echo ""
echo "============================================"
echo "Done. Results:"
echo "============================================"
column -t -s',' < "$RESULTS_CSV"
echo ""
echo "CSV: $RESULTS_CSV"
