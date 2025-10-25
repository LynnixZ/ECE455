# !/ usr / bin / env zsh
# SBATCH -- partition = instruction
# SBATCH -- time =00:01:00
# SBATCH -- ntasks =1
# SBATCH -- cpus - per - task =1
# SBATCH -- gpus - per - task =1
# SBATCH -- output = shared_warmup . output
cd $SLU RM_SU BMIT_D IR
module load nvidia / cuda
nvcc shared_warmup . cu -o shared_warmup
./ shared_warmup
