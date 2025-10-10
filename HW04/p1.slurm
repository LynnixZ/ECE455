#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=hello.output

set -euo pipefail
module purge
module load nvidia/cuda

nvcc -O2 hello.cu -o hello
srun ./hello
