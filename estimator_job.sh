#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name test_nn
#SBATCH --output=/scratch/r/rhlozek/rylan/slurm_output/%j.txt
#SBATCH --gres=gpu:4

cd $SLURM_SUBMIT_DIR

module load cuda/9.2 cudnn/cuda9.2/7.1.4 nccl/2.2.13 anaconda3
source activate frb

python /home/r/rhlozek/rylan/frby/estimator.py \
    --job-dir=/scratch/r/rhlozek/rylan/models/$SLURM_JOB_ID/

source deactivate
module purge

