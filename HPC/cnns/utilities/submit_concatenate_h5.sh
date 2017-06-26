#!/usr/bin/env bash
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_concatenate_h5.out -e /home/woody/capn/mppi033h/logs/submit_concatenate_h5.err

# submit script for the concatenate_h5 tool.
# submit with ' qsub -l nodes=1:ppn=4,walltime=01:00:00 submit_concatenate_h5.sh'

projection_path=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz
input_list_name=muon-CC_and_elec-NC_xyz.list
output_name=muon-CC_and_elec-NC_each_600_xyz.h5
chunksize=32


time python concatenate_h5.py --list ${projection_path}/${input_list_name} --chunksize 32 ${projection_path}/concatenated/${output_name}
