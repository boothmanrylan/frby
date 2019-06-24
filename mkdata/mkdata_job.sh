#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=6:00:00
#SBATCH --job-name mkdata
#SBATCH --output=/scratch/r/rhlozek/rylan/slurm_output/%j
#SBATCH --signal=SIGINT@120

# designed to run on niagara not SOSCIP GPU

cd $SLURM_SUBMIT_DIR

module purge
module load intelpython3/2019u3
module load intel/2019u3
module load intelmpi/2019u3

source /home/r/rhlozek/rylan/.virtualenvs/frb/bin/activate

export OMP_NUM_THREADS=1

mpirun python /home/r/rhlozek/rylan/frby/mkdata/mkdata.py

deactivate

module purge
