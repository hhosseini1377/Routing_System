# Routing System

A distributed LLM serving system that routes requests between multiple model backends using vLLM and FastAPI.

## Overview

The system consists of:
- **Router** (port 8000): Routes requests randomly between two model backends
- **Model A** (port 8001): First model backend with configurable GPU memory
- **Model B** (port 8002): Second model backend with configurable GPU memory

Both model servers use vLLM with tensor parallelism across 2 GPUs, and NVIDIA MPS for GPU sharing.

## Quick Start

1. Configure models and ports in `scripts/config.sh`

2. Start all services:
```bash
./scripts/start_services.sh
```

3. Send a request:
```bash
curl -N -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "temperature": 0.7, "max_tokens": 256}'
```

4. Test with multiple requests:
```bash
./scripts/send_requests.sh
```

## Configuration

Edit `scripts/config.sh` to set:
- Model names (HuggingFace IDs)
- GPU memory utilization for each backend
- Maximum sequence lengths
- Port numbers

## Requirements

- Python 3.11+
- 2 CUDA-capable GPUs
- vLLM, FastAPI, httpx
