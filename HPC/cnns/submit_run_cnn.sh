#!/usr/bin/env bash -l
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_run_cnn_${PBS_JOBID}.out -e /home/woody/capn/mppi033h/logs/submit_run_cnn_${PBS_JOBID}.err

# run with 'qsub.tinygpu -l walltime=23:59:00,nodes=1:ppn=4:gtx1080 submit_run_cnn.sh'

source /home/hpc/capn/mppi033h/.bash_profile

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/cnns
cd ${CodeFolder}
source env_cnn.sh

python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/h5/xyzt/concatenated/train_muon-CC_and_elec-CC_each_480_xyzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/h5/xyzt/concatenated/test_muon-CC_and_elec-CC_each_120_xyzt_shuffled.h5
