#!/bin/bash

# Load required modules (needed for Python 3.11.3 shared libraries)
module load GCCcore/11.3.0
module load Python/3.11.3

source ./env/bin/activate

# Change the cache directory for huggingface

echo "Sending requests concurrently with 100 requests..."
python -m send_requests concurrent 1000