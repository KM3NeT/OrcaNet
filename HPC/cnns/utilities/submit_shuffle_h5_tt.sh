#!/usr/bin/env bash
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_shuffle_h5_$PBS_JOBID.out -e /home/woody/capn/mppi033h/logs/submit_shuffle_h5_$PBS_JOBID.err

# submit script for the shuffle_h5 tool. Specific version for shuffling the train and test datasets (_tt).
# submit with 'qsub -l nodes=1:ppn=4:sl32g,walltime=01:01:00 submit_shuffle_h5_tt.sh'


projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated
input_filename_train=train_muon-CC_and_elec-CC_each_240_xzt.h5
input_filename_test=test_muon-CC_and_elec-CC_each_60_xzt.h5
#input_filename_test=test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5
chunksize=32

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/cnns/utilities
cd ${CodeFolder}

# no parallel shuffling, since we are limited by RAM
time python shuffle_h5.py -d --chunksize ${chunksize} ${projection_path}/${input_filename_train} > ${projection_path}/logs/cout/${input_filename_train}.txt
wait
time python shuffle_h5.py -d --chunksize ${chunksize} ${projection_path}/${input_filename_test} > ${projection_path}/logs/cout/${input_filename_test}.txt
