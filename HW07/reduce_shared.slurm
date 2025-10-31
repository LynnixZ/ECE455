# !/ usr / bin / env zsh
# SBATCH -- partition = instruction
# SBATCH -- time =00:03:00
# SBATCH -- ntasks =1
# SBATCH -- cpus - per - task =1
# SBATCH -- gpus - per - task =1
# SBATCH -- output = reduce_shared . output
cd $SLU RM_SU BMIT_D IR
module load nvidia / cuda
nvcc reduce_shared . cu -o reduce_shared
./ reduce_shared
