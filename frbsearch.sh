#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=08:00:00
#SBATCH --job-name frbsearch
#SBATCH --output=/scratch/r/rhlozek/rylan/slurm_output/%j.txt
#SBATCH --gres=gpu:4

. /etc/profile.d/modules.sh

module purge
module load cuda/9.2 cudnn/cuda9.2/7.5.0 nccl/2.4.2 anaconda3

source activate astro

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR

python /home/r/rhlozek/rylan/frby/model.py

conda deactivate
module purge

