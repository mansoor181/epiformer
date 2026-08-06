#!/bin/bash

# Multi-GPU WandB sweep runner with multiple agents per GPU
# Usage:
#   ./scripts/sweeps/epiformer_sweep.sh --server local --gpu_count 2
#   ./scripts/sweeps/epiformer_sweep.sh --server local --gpu_count 2 --agents_per_gpu 3
#   ./scripts/sweeps/epiformer_sweep.sh --server local --gpu_count 2 --agents_per_gpu 2 --config scripts/sweeps/m1_noplm_sweep.yaml
#   ./scripts/sweeps/epiformer_sweep.sh --server local --sweep_id ENTITY/PROJECT/SWEEP_ID  # resume existing sweep

usage() {
    echo "Usage: $0 --server <name> [options]"
    echo ""
    echo "Options:"
    echo "  --server <name>           Server name (required)"
    echo "  --gpu_count <N>           Number of GPUs (default: 2)"
    echo "  --agents_per_gpu <N>      Agents per GPU (default: 1)"
    echo "  --config <path>           Sweep YAML config (default: scripts/sweeps/epiformer_sweep.yaml)"
    echo "  --sweep_id <id>           Resume an existing sweep instead of creating a new one"
    echo "  --project <name>          WandB project (default: m3epi_v3)"
    echo ""
    echo "Examples:"
    echo "  $0 --server local --gpu_count 2 --agents_per_gpu 3"
    echo "  $0 --server local --config scripts/sweeps/m1_noplm_sweep.yaml --gpu_count 2 --agents_per_gpu 2"
    echo "  $0 --server local --sweep_id user/m3epi_v3/abc123  # resume"
    exit 1
}

# Defaults
gpu_count=2
agents_per_gpu=3
sweep_config="scripts/sweeps/m1_noplm_sweep.yaml"
sweep_id=""
project="m3epi_v3"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --server) server="$2"; shift 2 ;;
        --gpu_count) gpu_count="$2"; shift 2 ;;
        --agents_per_gpu) agents_per_gpu="$2"; shift 2 ;;
        --config) sweep_config="$2"; shift 2 ;;
        --sweep_id) sweep_id="$2"; shift 2 ;;
        --project) project="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

if [ -z "$server" ]; then
    echo "Error: --server required"
    usage
fi

total_agents=$((gpu_count * agents_per_gpu))

echo "============================================"
echo "Multi-GPU WandB Sweep"
echo "============================================"
echo "Server:          $server"
echo "GPUs:            0-$((gpu_count-1))"
echo "Agents per GPU:  $agents_per_gpu"
echo "Total agents:    $total_agents"
echo "Sweep config:    $sweep_config"
echo "WandB project:   $project"
echo "============================================"

mkdir -p logs

# Initialize sweep or resume existing
if [ -n "$sweep_id" ]; then
    echo "Resuming existing sweep: $sweep_id"
else
    if [ ! -f "$sweep_config" ]; then
        echo "Error: Sweep config not found: $sweep_config"
        exit 1
    fi

    echo "Initializing WandB sweep..."
    sweep_output=$(wandb sweep --project "$project" "$sweep_config" 2>&1)
    sweep_id=$(echo "$sweep_output" | grep -o 'wandb agent [^[:space:]]*' | awk '{print $3}')

    if [ -z "$sweep_id" ]; then
        echo "Failed to initialize sweep. Output:"
        echo "$sweep_output"
        exit 1
    fi
fi

echo "Sweep ID: $sweep_id"
echo ""

# Launch agents: multiple per GPU
pids=()
for gpu_id in $(seq 0 $((gpu_count-1))); do
    for agent_num in $(seq 1 $agents_per_gpu); do
        log_file="logs/sweep_${server}_gpu${gpu_id}_agent${agent_num}.log"
        echo "Starting agent $agent_num on GPU $gpu_id -> $log_file"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup wandb agent "$sweep_id" > "$log_file" 2>&1 &
        pid=$!
        pids+=($pid)
        echo "  PID: $pid"
        sleep 3  # Stagger startup to avoid race conditions
    done
done

echo ""
echo "============================================"
echo "All $total_agents agents started"
echo "============================================"
echo "Sweep ID: $sweep_id"
echo "PIDs: ${pids[*]}"
echo ""
echo "Monitor logs:"
for gpu_id in $(seq 0 $((gpu_count-1))); do
    for agent_num in $(seq 1 $agents_per_gpu); do
        echo "  GPU $gpu_id Agent $agent_num: tail -f logs/sweep_${server}_gpu${gpu_id}_agent${agent_num}.log"
    done
done
echo ""
echo "Kill all agents: kill ${pids[*]}"
echo "Or use: pkill -f 'wandb agent'"
