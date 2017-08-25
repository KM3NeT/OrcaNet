#!/usr/bin/env bash

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/cnns
cd ${CodeFolder}
activate_env1
source env_cnn.sh

python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5
