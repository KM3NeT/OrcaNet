#!/usr/bin/env bash
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_concatenate_h5_$PBS_JOBID.out -e /home/woody/capn/mppi033h/logs/submit_concatenate_h5_$PBS_JOBID.err

# submit script for the concatenate_h5 tool. Specific version for concatenating the single files to train and test datasets (_tt).
# submit with 'qsub -l nodes=1:ppn=4,walltime=01:01:00 submit_concatenate_h5_tt.sh'

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/cnns/utilities
projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt

input_list_name_train=muon-CC_and_elec-CC_xzt_1_to_240.list # train
output_name_train=train_muon-CC_and_elec-CC_each_240_xzt.h5 # train
input_list_name_test=muon-CC_and_elec-CC_xzt_241_to_300.list # test
output_name_test=test_muon-CC_and_elec-CC_each_60_xzt.h5 # test

chunksize=32

cd ${CodeFolder}

(time taskset -c 0 python concatenate_h5.py --list ${projection_path}/${input_list_name_train} --chunksize ${chunksize} ${projection_path}/concatenated/${output_name_train} > ${projection_path}/logs/cout/${output_name_train}.txt) &
(time taskset -c 1 python concatenate_h5.py --list ${projection_path}/${input_list_name_test} --chunksize ${chunksize} ${projection_path}/concatenated/${output_name_test} > ${projection_path}/logs/cout/${output_name_test}.txt)
wait
