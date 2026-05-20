#!/bin/bash

# Multi-GPU WandB sweep runner for test.sh configuration
# Usage: ./scripts/sweeps/epiformer_sweep.sh --server amai --gpu_count 3

usage() {
    echo "Usage: $0 --server <name> [--gpu_count <count>]"
    echo "Example: $0 --server amai --gpu_count 3"
    echo "Default GPU count: 3"
    exit 1
}

# Parse arguments
gpu_count=3  # Default to 3 GPUs
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --server) server="$2"; shift 2 ;;
        --gpu_count) gpu_count="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

if [ -z "$server" ]; then
    echo "Error: --server required"
    usage
fi

sweep_config="scripts/sweeps/epiformer_sweep.yaml"

echo "Starting multi-GPU hyperparameter sweep"
echo "Server: $server | GPUs: 0-$((gpu_count-1))"

mkdir -p logs

# Initialize single sweep (all agents will connect to this)
echo "Initializing WandB sweep..."
sweep_output=$(wandb sweep --project m3epi_v3 "$sweep_config" 2>&1)
sweep_id=$(echo "$sweep_output" | grep -o 'wandb agent [^[:space:]]*' | awk '{print $3}')

if [ -z "$sweep_id" ]; then
    echo "Failed to initialize sweep. Output:"
    echo "$sweep_output"
    exit 1
fi

echo "Sweep ID: $sweep_id"

# Launch agents on each GPU
pids=()
for gpu_id in $(seq 0 $((gpu_count-1))); do
    echo "Starting agent on GPU $gpu_id..."
    export CUDA_VISIBLE_DEVICES=$gpu_id
    nohup wandb agent "$sweep_id" > "logs/sweep_${server}_gpu${gpu_id}.log" 2>&1 &
    pid=$!
    pids+=($pid)
    echo "GPU $gpu_id agent started (PID: $pid)"
    sleep 2  # Stagger startup
done

echo ""
echo "All agents started successfully!"
echo "Sweep ID: $sweep_id"
echo "PIDs: ${pids[*]}"
echo ""
echo "Monitor logs:"
for gpu_id in $(seq 0 $((gpu_count-1))); do
    echo "  GPU $gpu_id: tail -f logs/sweep_${server}_gpu${gpu_id}.log"
done
echo ""
echo "Kill all agents: kill ${pids[*]}"
echo "Or use: pkill -f 'wandb agent'"