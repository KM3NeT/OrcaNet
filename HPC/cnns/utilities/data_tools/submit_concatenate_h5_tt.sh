#!/usr/bin/env bash
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_concatenate_h5_${PBS_JOBID}.out -e /home/woody/capn/mppi033h/logs/submit_concatenate_h5_${PBS_JOBID}.err

# submit script for the concatenate_h5 tool. Specific version for concatenating the single files to train and test datasets (_tt).
# submit with 'qsub -l nodes=1:ppn=4,walltime=01:01:00 submit_concatenate_h5_tt.sh'
# Make a .txt file with < find /path/to/files -name "file_x-*.h5" | sort --version-sort > listname.list >
# Don't forget to create the logs/cout folder in the projection_path!

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/cnns/utilities/data_tools
cd ${CodeFolder}

chunksize=32

#------ 3-100GeV ------#
# 3d - xzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt
#input_list_name_train=muon-CC_and_elec-CC_xzt_1_to_240.list # train
#output_name_train=train_muon-CC_and_elec-CC_each_240_xzt.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_xzt_241_to_300.list # test
#output_name_test=test_muon-CC_and_elec-CC_each_60_xzt.h5 # test
#----batch 2
#input_list_name_train=muon-CC_and_elec-CC_xzt_301_to_540.list # train
#output_name_train=train_muon-CC_and_elec-CC_each_240_batch_301-540_xzt.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_xzt_541_to_600.list # test
#output_name_test=test_muon-CC_and_elec-CC_each_60_batch_541-600_xzt.h5 # test

# 3d - yzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/yzt
#input_list_name_train=muon-CC_and_elec-CC_yzt_1_to_240.list # train
#output_name_train=train_muon-CC_and_elec-CC_each_240_yzt.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_yzt_241_to_300.list # test
#output_name_test=test_muon-CC_and_elec-CC_each_60_yzt.h5 # test


# 4d - xyzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/h5/xyzt
#input_list_name_train=muon-CC_and_elec-CC_xyzt_1_to_480.list # train
#output_name_train=train_muon-CC_and_elec-CC_each_480_xyzt.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_xyzt_481_to_600.list # test
#output_name_test=test_muon-CC_and_elec-CC_each_120_xyzt.h5 # test
#compression=--compression

# 4d - xyzt - with run_id
projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/with_run_id/h5/xyzt
input_list_name_train=muon-CC_and_elec-CC_xyzt_1_to_480.list # train
output_name_train=train_muon-CC_and_elec-CC_each_480_xyzt.h5 # train
input_list_name_test=muon-CC_and_elec-CC_xyzt_481_to_600.list # test
output_name_test=test_muon-CC_and_elec-CC_each_120_xyzt.h5 # test
compression=--compression



#------ 10-100GeV ------#
# 2d - yz
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/yz
#input_list_name_train=muon-CC_and_elec-CC_10-100GeV_yz_1_to_480.list # train
#output_name_train=train_muon-CC_and_elec-CC_10-100GeV_each_480_yz.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_10-100GeV_yz_481_to_600.list # test
#output_name_test=test_muon-CC_and_elec-CC_10-100GeV_each_120_yz.h5 # test
# 2d - zt - muon-CC only
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/zt
#input_list_name_train=muon-CC_10-100GeV_zt_1_to_480.list # train
#output_name_train=train_muon-CC_10-100GeV_each_480_zt.h5 # train
#input_list_name_test=muon-CC_10-100GeV_zt_481_to_600.list # test
#output_name_test=test_muon-CC_10-100GeV_each_120_zt.h5 # test
# 3d - xyz
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo3d/h5/xyz
#input_list_name_train=muon-CC_and_elec-CC_10-100GeV_xyz_1_to_480.list # train
#output_name_train=train_muon-CC_and_elec-CC_10-100GeV_each_480_xyz.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_10-100GeV_xyz_481_to_600.list # test
#output_name_test=test_muon-CC_and_elec-CC_10-100GeV_each_120_xyz.h5 # test
# 3d - yzt
#projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo3d/h5/yzt
#input_list_name_train=muon-CC_and_elec-CC_10-100GeV_yzt_1_to_480.list # train
#output_name_train=train_muon-CC_and_elec-CC_10-100GeV_each_480_yzt.h5 # train
#input_list_name_test=muon-CC_and_elec-CC_10-100GeV_yzt_481_to_600.list # test
#output_name_test=test_muon-CC_and_elec-CC_10-100GeV_each_120_yzt.h5 # test




(time taskset -c 0 python concatenate_h5.py --list ${projection_path}/${input_list_name_train} ${compression} --chunksize ${chunksize} ${projection_path}/concatenated/${output_name_train} > ${projection_path}/logs/cout/${output_name_train}.txt) &
(time taskset -c 1 python concatenate_h5.py --list ${projection_path}/${input_list_name_test} ${compression} --chunksize ${chunksize} ${projection_path}/concatenated/${output_name_test} > ${projection_path}/logs/cout/${output_name_test}.txt)
wait
