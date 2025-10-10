#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=vector_add_streams.output

set -euo pipefail
module purge
module load nvidia/cuda

nvcc -O2 vector_add_streams.cu -o vector_add_streams
srun ./vector_add_streams
