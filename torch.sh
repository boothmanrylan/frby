#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=04:00:00
#SBATCH --job-name horovod
#SBATCH --output=/scratch/r/rhlozek/rylan/slurm_output/%j.txt
#SBATCH --gres=gpu:4

. /etc/profile.d/modules.sh

module load cuda/10.1 nccl/cuda10.1/2.4.2 anaconda3
source activate pytorch

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR

nodelist=$SCRATCH/horovod/slurm-nodelist
rankfile=$SCRATCH/horovod/rankfile
checkpoint=$SCRATCH/models/$SLURM_JOB_ID/

# mkdir $checkpoint

scontrol show hostnames > $nodelist
python $SCRATCH/horovod/createrankfile.py $nodelist > $rankfile

mpirun -np $((SLURM_NTASKS/5)) -bind-to hwthread -rf $rankfile \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x OMP_NUM_THREADS -x HOROVOD_MPI_THREADS_DISABLE=1 \
    python $HOME/frby/torch_model.py --checkpoint $SCRATCH/models/34290/
