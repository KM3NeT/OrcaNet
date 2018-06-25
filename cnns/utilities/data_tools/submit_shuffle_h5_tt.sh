#!/usr/bin/env bash
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_shuffle_h5_${PBS_JOBID}.out -e /home/woody/capn/mppi033h/logs/submit_shuffle_h5_${PBS_JOBID}.err

# submit script for the shuffle_h5 tool. Specific version for shuffling the train and test datasets (_tt).
# submit with 'qsub -l nodes=1:ppn=4:sl32g,walltime=01:01:00 submit_shuffle_h5_tt.sh'
# Don't forget to create the logs/cout folder in the projection_path (concatenated/logs/cout)!

CodeFolder=/home/woody/capn/mppi033h/Code/OrcaNet/cnns/utilities/data_tools
cd ${CodeFolder}
source activate /home/hpc/capn/mppi033h/.virtualenv/h5_to_histo_env/

chunksize=32


#------ 3-100GeV ------#
# 3d - xzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated
#input_filename_train=train_muon-CC_and_elec-CC_each_240_xzt.h5
#input_filename_test=test_muon-CC_and_elec-CC_each_60_xzt.h5
#---- batch 2
#input_filename_train=train_muon-CC_and_elec-CC_each_240_batch_301-540_xzt.h5
#input_filename_test=test_muon-CC_and_elec-CC_each_60_batch_541-600_xzt.h5

# 3d - yzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/yzt/concatenated
#input_filename_train=train_muon-CC_and_elec-CC_each_240_yzt.h5
#input_filename_test=test_muon-CC_and_elec-CC_each_60_yzt.h5

#4d - xyzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/h5/xyzt/concatenated
#input_filename_train=train_muon-CC_and_elec-CC_each_480_xyzt.h5
#input_filename_test=test_muon-CC_and_elec-CC_each_120_xyzt.h5

# Notes for shuffling 4d on TinyFat:
# qsub.tinyfat -I -q broadwell512 -l walltime=02:59:00,nodes=1:ppn=56
# python shuffle_h5.py --chunksize 32 --compression --n_shuffles 20 path/to/file
#
# I.e. python shuffle_h5.py --chunksize 32 --compression --n_shuffles 20 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/with_run_id/h5/xyzt/concatenated/train_muon-CC_and_elec-CC_each_480_xyzt_shuffled_1.h5

#------ 10-100GeV ------#
# 2d - yz
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/yz/concatenated
#input_filename_train=train_muon-CC_and_elec-CC_10-100GeV_each_480_yz.h5
#input_filename_test=test_muon-CC_and_elec-CC_10-100GeV_each_120_yz.h5
# 2d - zt - muon-CC only
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/zt/concatenated
#input_filename_train=train_muon-CC_10-100GeV_each_480_zt.h5
#input_filename_test=test_muon-CC_10-100GeV_each_120_zt.h5
# 3d - xyz
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo3d/h5/xyz/concatenated
#input_filename_train=train_muon-CC_and_elec-CC_10-100GeV_each_480_xyz.h5
#input_filename_test=test_muon-CC_and_elec-CC_10-100GeV_each_120_xyz.h5
# 3d - yzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo3d/h5/yzt/concatenated
#input_filename_train=train_muon-CC_and_elec-CC_10-100GeV_each_480_yzt.h5
#input_filename_test=test_muon-CC_and_elec-CC_10-100GeV_each_120_yzt.h5


# no parallel shuffling, since we are limited by RAM
##time python shuffle_h5.py -d --chunksize ${chunksize} ${projection_path}/${input_filename_train} > ${projection_path}/logs/cout/${input_filename_train}.txt
##wait
##time python shuffle_h5.py -d --chunksize ${chunksize} ${projection_path}/${input_filename_test} > ${projection_path}/logs/cout/${input_filename_test}.txt


