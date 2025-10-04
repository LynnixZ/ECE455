# p3.slurm
#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=parallel_sum_timing.output

cd "$SLURM_SUBMIT_DIR"

# module load gcc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=true
export OMP_PLACES=cores

g++ -O3 -march=native -fopenmp parallel_sum_timing.cpp -o parallel_sum_timing
./parallel_sum_timing
