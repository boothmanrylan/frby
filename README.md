# frby

In order to run estimator.py on the SOSCIP GPU cluster you will need a conda
environment:
* `module load anaconda3`
* `conda create --name ENV_NAME --file environment-spec.txt`

To use estimator_job.sh with the environment you need to also change line 12
of estimator_job.sh to: `source activate ENV_NAME`

You will also need load the following modules:
* `cuda/9.2`
* `cudnn/cuda9.2/7.1.4`
* `nccl/2.2.13`


## File descriptions

estimator.py --> The classifier

estimator_job.sh --> The script to use for job submission i.e. `sbatch
estimator_job.sh` for an estimator/classification job.
for more information on the `#SBATCH` commands see
https://docs.scinet.utoronto.ca/index.php/SOSCIP_GPU#Job_Submission

mkdata_job.sh --> The script to use for job submission for a mkdata job. It is
possible that this script is broken. As well, estimator_job.sh is designed to
run on the SOSCIP GPU cluster, but mkdata_job.sh is designed to run on niagara

evironment-spec.txt --> Conda environment specification

mkdata/ --> Everything needed to create the simulated dataset

mkdata/frb.py --> Class to simulate fast radio bursts, based on https://github.com/liamconnor/single_pulse_ml which is in turn based on https://github.com/liamconnor/single_pulse_ml  

mkdata/psr.py --> Class to simulate pulsars, uses the class in frb.py to make
the simulations

mkdata/rfi.py --> Classes to simulate radio frequency interference

mkdata/tools.py --> Method to read single/multiple VDIF files and bin/reduce
them in time.

mkdata/mkdata.py --> Script to generate the simulated dataset using the other
files in mkdata/

mkdata/modifications.py --> Holds a list containing the parameters to use when
create the rfi examples

mkdata/tensorflow_records.py --> Converts the .npy files created by
mkdata/mkdata.py into Tensorflow .tfrecords files 


