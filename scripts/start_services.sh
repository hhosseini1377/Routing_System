#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Function to get PIDs of processes listening on a port
get_port_pids() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti ":$port" 2>/dev/null || true
    elif command -v fuser >/dev/null 2>&1; then
        fuser "$port/tcp" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i!=":") print $i}' || true
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :$port" 2>/dev/null | awk '/pid=/ {match($0, /pid=([0-9]+)/, a); print a[1]}' | sort -u || true
    fi
}

# Cleanup function to kill all child processes and processes on ports
cleanup() {
    local exit_code=$?
    echo "Cleaning up services..." >&2
    
    # Kill all background processes if they exist
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid" >&2
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Kill processes listening on the service ports (safety measure)
    local ports=("${ROUTER_PORT}" "${UVICORN_MODEL_A_PORT}" "${UVICORN_MODEL_B_PORT}")
    for port in "${ports[@]}"; do
        local port_pids
        port_pids=$(get_port_pids "$port")
        
        if [ -n "$port_pids" ]; then
            for port_pid in $port_pids; do
                # Skip if we already handled this PID
                local already_handled=false
                for existing_pid in "${PIDS[@]}"; do
                    if [ "$port_pid" = "$existing_pid" ]; then
                        already_handled=true
                        break
                    fi
                done
                if [ "$already_handled" = "false" ] && kill -0 "$port_pid" 2>/dev/null; then
                    echo "Killing process $port_pid on port $port" >&2
                    kill "$port_pid" 2>/dev/null || true
                fi
            done
        fi
    done
    
    # Wait a bit for graceful shutdown
    sleep 2
    
    # Force kill if still running (PIDs)
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    # Force kill processes still on ports
    for port in "${ports[@]}"; do
        local port_pids
        port_pids=$(get_port_pids "$port")
        
        if [ -n "$port_pids" ]; then
            for port_pid in $port_pids; do
                if kill -0 "$port_pid" 2>/dev/null; then
                    echo "Force killing process $port_pid on port $port" >&2
                    kill -9 "$port_pid" 2>/dev/null || true
                fi
            done
        fi
    done
    
    exit $exit_code
}

trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES="0,1"

# Configure NVIDIA MPS for multi-process GPU sharing
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps 
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un) 

# Change the cache directory for huggingface
export TRITON_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/triton
export XDG_CONFIG_HOME=/data/gpfs/projects/punim2662/.config
export VLLM_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/vllm
export VLLM_CACHE_ROOT=/data/gpfs/projects/punim2662/.cache/vllm
export TORCH_HOME=/data/gpfs/projects/punim2662/.cache/torch/
export TORCHINDUCTOR_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/torch/inductor
export CUDA_CACHE_PATH=/data/gpfs/projects/punim2662/.cache/nvidia/
export HF_HOME=/data/gpfs/projects/punim2662/.cache/huggingface

# Ensure MPS is active (optional but recommended when sharing GPUs).
if ! "${ROOT_DIR}/scripts/start_mps.sh"; then
    echo "ERROR: Failed to start MPS" >&2
    exit 1
fi

if ! source scripts/config.sh; then
    echo "ERROR: Failed to source config.sh" >&2
    exit 1
fi

# Array to track PIDs
declare -a PIDS
declare -a SERVICE_NAMES

# Function to wait for a service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://127.0.0.1:${port}/docs" >/dev/null 2>&1 || \
           curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || \
           netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            echo "$service_name is ready on port $port"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "WARNING: $service_name may not be ready after $max_attempts seconds" >&2
    return 1
}

# Function to start a service and verify it's running
start_service() {
    local service_name=$1
    local start_cmd=$2
    local port=$3
    
    echo "Starting $service_name..."
    eval "$start_cmd" &
    local pid=$!
    
    # Check if process started successfully
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: $service_name failed to start (PID $pid died immediately)" >&2
        return 1
    fi
    
    PIDS+=("$pid")
    SERVICE_NAMES+=("$service_name")
    echo "$service_name started with PID $pid"
    
    # Wait for service to be ready (non-blocking, continue anyway)
    wait_for_service "$port" "$service_name" || true
    
    return 0
}

# Router
ROUTER_CMD="MODEL_A_URL=\"http://127.0.0.1:${UVICORN_MODEL_A_PORT}\" \
MODEL_B_URL=\"http://127.0.0.1:${UVICORN_MODEL_B_PORT}\" \
uvicorn router:app --host 0.0.0.0 --port ${ROUTER_PORT}"

if ! start_service "Router" "$ROUTER_CMD" "${ROUTER_PORT}"; then
    echo "ERROR: Failed to start Router" >&2
    exit 1
fi
PID_ROUTER="${PIDS[-1]}"

# Start model B (Large Model)
MODEL_B_CMD="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${MODEL_B_ACTIVE_THREAD_PERCENTAGE} \
MODEL_NAME=\"${MODEL_B_NAME}\" \
TENSOR_PARALLEL_SIZE=2 \
GPU_MEMORY_UTILIZATION=${MODEL_B_GPU_MEMORY_UTILIZATION} \
MAX_MODEL_LEN=${MODEL_B_MAX_LEN} \
UVICORN_PORT=${UVICORN_MODEL_B_PORT} \
uvicorn model_server:app --host 0.0.0.0 --port ${UVICORN_MODEL_B_PORT}"

if ! start_service "Model B" "$MODEL_B_CMD" "${UVICORN_MODEL_B_PORT}"; then
    echo "ERROR: Failed to start Model B" >&2
    exit 1
fi
PID_1B="${PIDS[-1]}"

# Start model A (Small Model)
MODEL_A_CMD="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${MODEL_A_ACTIVE_THREAD_PERCENTAGE} \
MODEL_NAME=\"${MODEL_A_NAME}\" \
TENSOR_PARALLEL_SIZE=2 \
GPU_MEMORY_UTILIZATION=${MODEL_A_GPU_MEMORY_UTILIZATION} \
MAX_MODEL_LEN=${MODEL_A_MAX_LEN} \
UVICORN_PORT=${UVICORN_MODEL_A_PORT} \
uvicorn model_server:app --host 0.0.0.0 --port ${UVICORN_MODEL_A_PORT}"

if ! start_service "Model A" "$MODEL_A_CMD" "${UVICORN_MODEL_A_PORT}"; then
    echo "ERROR: Failed to start Model A" >&2
    exit 1
fi
PID_8B="${PIDS[-1]}"

echo "All services started:"
echo "  Router on :${ROUTER_PORT} (PID: $PID_ROUTER)"
echo "  Model A (${MODEL_A_NAME}) on :${UVICORN_MODEL_A_PORT} (PID: $PID_8B)"
echo "  Model B (${MODEL_B_NAME}) on :${UVICORN_MODEL_B_PORT} (PID: $PID_1B)"

# Wait for any process to exit and identify which one
wait -n "${PID_8B}" "${PID_1B}" "${PID_ROUTER}"
exit_code=$?

# Identify which process exited
for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        echo "ERROR: ${SERVICE_NAMES[$i]} (PID ${PIDS[$i]}) exited with code $exit_code" >&2
        break
    fi
done

exit $exit_code
