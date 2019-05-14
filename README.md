# frby

In order to run model.py on the SOSCIP GPU cluster:

1. Create a conda environment
    1. Unload any previously loaded modules: `module purge`
    1. Load required modules: `module load cuda/9.2 cudnn/cuda/9.2/7.5.0 nccl/2.4.2 anaconda3`
    1. Create the environment: `conda create -n <ENV_NAME> python=3.6`
    1. Activate the environment: `source activate <ENV_NAME>`
    1. Install necessary packages: `conda install -n <ENV_NAME>
      keras-applications==1.0.6 keras-preprocessing==1.0.5 scipy mock cython
      numpy protobuf grpcio markdown html5lib werkzeug absl-py bleach six
      openblas h5py astor gast  setuptools scikit-image`
    1. Install Tensorflow: `pip install
      /scinet/sgc/Applications/TensorFlow_wheels/conda/tensorflow-1.13.1-cp36-cp36m-linux_ppc64le.whl`
    1. Verify installation: `python -c "import tensorflow as tf;
      tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000,
      1000])))"`
    1. Leave environment: `conda deactivate; module purge`
1. Modify frbsearch.sh
    * Change line 6 so the output points to a file in your scratch directory
    * Change line 14 from `source activate astro` to `source activate <ENV_NAME>`
    * Change line 20 to the path to where model.py is located in your home directory.
1. Submit job: `sbatch frbsearch.sh`

## File descriptions

model.py --> The classifier

frbsearch.sh --> Job submission script i.e. `sbatch frbsearch.sh`

mkdata_job.sh --> The script to use for job submission for a mkdata job. It is
possible that this script is broken. As well, frbsearch.sh is designed to
run on the SOSCIP GPU cluster, but mkdata_job.sh is designed to run on niagara

evironment-spec.txt --> Conda environment specification

mkdata/ --> Everything needed to create the simulated dataset

mkdata/frb.py --> Class to simulate fast radio bursts, based on
https://github.com/liamconnor/single_pulse_ml which is in turn based on
https://github.com/kiyo-masui/burst_search

mkdata/psr.py --> Class to simulate pulsars, uses the class in frb.py to make
the simulations

mkdata/rfi.py --> Classes to simulate radio frequency interference

mkdata/tools.py --> Method to read single/multiple VDIF files and bin/reduce
them in time.

mkdata/mkdata.py --> Script to generate the simulated dataset using the other
files in mkdata/

mkdata/modifications.py --> Holds a list containing the parameters to use when
creating the rfi examples

mkdata/build_tfrecords.py --> Converts the .npy files created by
mkdata/mkdata.py into TensorFlowRecords


