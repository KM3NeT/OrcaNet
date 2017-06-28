#!/usr/bin/env bash
#
#PBS -o /home/woody/capn/mppi033h/logs/submit_shuffle_h5_$PBS_JOBID.out -e /home/woody/capn/mppi033h/logs/submit_shuffle_h5_$PBS_JOBID.err

# submit script for the shuffle_h5 tool.
# submit with ' qsub -l nodes=1:ppn=4,walltime=01:00:00 submit_shuffle_h5.sh'

input_path_file=
chunksize=32

time python shuffle_h5.py -d --chunksize ${chunksize} ${input_path_file}