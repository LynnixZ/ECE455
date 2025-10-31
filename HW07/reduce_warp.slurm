# !/ usr / bin / env zsh
# SBATCH -- partition = instruction
# SBATCH -- time =00:03:00
# SBATCH -- ntasks =1
# SBATCH -- cpus - per - task =1
# SBATCH -- gpus - per - task =1
# SBATCH -- output = reduce_warp . output
cd $SLU RM_SU BMIT_D IR
module load nvidia / cuda
nvcc reduce_warp . cu -o reduce_warp
./ reduce_warp
