#!/usr/bin/env bash
# remember: qsub.tinygpu -I -l walltime=00:59:00,nodes=1:ppn=4:gtx1080

# load conda virtualenv
source activate /home/hpc/capn/mppi033h/.virtualenv/h5_to_histo_env/
# required:
module use -a /home/vault/capn/shared/apps/U16/modules
module load cudnn/6.0-cuda8.0
# obtaining software from the outside world requires a working proxy to be set
export https_proxy=https://pi4060.physik.uni-erlangen.de:8888
export http_proxy=http://pi4060.physik.uni-erlangen.de:8888
pip install --user --upgrade tensorflow-gpu keras h5py numpy sklearn
#pip install --upgrade tensorflow-gpu keras h5py numpy sklearn