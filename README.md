# frby

environment-spec.txt --> Conda file containing the packages needed to create
a clone of an environment that can run mkdata.py and estimator.py Create the
environment with
conda create --name THE_ENVIRONMENT_NAME --file environment-spec.txt

estimator.py --> The classifier

job.script --> The script to use for job submission i.e. `sbatch job.script`

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


