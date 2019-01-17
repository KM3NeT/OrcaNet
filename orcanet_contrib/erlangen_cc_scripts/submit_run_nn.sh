#!/bin/bash -l
#
# job name for PBS, out and error files will also have this name + an id
#PBS -N vgg3_test
#
# first non-empty non-comment line ends PBS options

cd $WOODYHOME/Code/work/

model="configs/models/example_model_2.toml"
list="configs/lists/example_list.list"
folder="trained_models/"

#Setup environment
. env_cnn.sh
#Execute training
python ../OrcaNet/orcanet/run_nn.py $model $list $folder


