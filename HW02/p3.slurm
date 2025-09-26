#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=race_conditions.output

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

g++ -O2 -std=c++17 race_conditions.cpp -o race_conditions -pthread
./race_conditions
