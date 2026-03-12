#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=deeplearn
#SBATCH --qos=deeplearngpgpu
#SBATCH --constraint=glb5
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00 
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH -A punim2662 
#SBATCH --output=outputs/jupyter_%j.out

# --- 1. Load Cluster Modules ---
# Load required modules (needed for Python 3.11.3 shared libraries)
module purge

module load GCCcore/11.3.0
module load Python/3.11.3
module load CUDA/12.4.1

# Set CUDA_HOME and PATH and LD_LIBRARY_PATH
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- 2. Activate the Virtual Environment ---
# Your 'env' folder must already exist on your file system
source env/bin/activate

# --- 3. Install/Update Packages and Register Kernel (Crucial Step) ---
# Running this ensures ipykernel is available and the kernel is registered
# It's okay to run 'pip install' every time; it will be skipped if satisfied.
pip install ipykernel jupyterlab peft # Ensure all required packages are installed/updated

# Register the environment as a kernel. It will overwrite the registration if it exists.
python -m ipykernel install --user --name=env --display-name "SLURM Job Kernel (PEFT)"

# --- 4. Start the Jupyter Server ---
# The server will now be running inside the activated 'env' and will have access to the registered kernel
jupyter lab --no-browser --port=8890 --ip=0.0.0.0

# --- Cleanup (Optional, but good practice) ---
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s