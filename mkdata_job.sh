#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=320
#SBATCH --time=6:00:00
#SBATCH --job-name mkdata
#SBATCH --output=/scratch/r/rhlozek/rylan/sbatch/sbatch_output_%j.txt
#SBATCH --signal=SIGINT 600


# designed to run on niagara not SOSCIP GPU

cd $SLURM_SUBMIT_DIR

module purge
module load intelpython3/2018.2
module load intel/2018.2
module load intelmpi/2018.2

source activate frb

export OMP_NUM_THREADS=1

mpirun python /home/r/rhlozek/rylan/frby/mkdata/mkdata.py
