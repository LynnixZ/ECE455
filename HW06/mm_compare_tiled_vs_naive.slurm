# !/ usr / bin / env zsh
# SBATCH -- partition = instruction
# SBATCH -- time =00:01:00
# SBATCH -- ntasks =1
# SBATCH -- cpus - per - task =1
# SBATCH -- gpus - per - task =1
# SBATCH -- output = m m _ c o m p a r e _ t i l e d _ v s _ n a i v e . output
cd $SLU RM_SU BMIT_D IR
module load nvidia / cuda
nvcc m m _ c o m p a r e _ t i l e d _ v s _ n a i v e . cu -o m m _ c o m p a r e _ t i l e d _ v s _ n a i v e
./ m m _ c o m p a r e _ t i l e d _ v s _ n a i v e
