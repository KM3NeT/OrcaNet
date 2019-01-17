#!/usr/bin/env bash
# run with 'sh submit_concatenate_and_shuffle_h5.sh'
CodeFolder=/home/woody/capn/mppi033h/Code/HPC/cnns/utilities/data_tools
cd ${CodeFolder}

submit_concatenate=$(qsub -l nodes=1:ppn=4,walltime=01:01:00 submit_concatenate_h5_tt.sh)
echo ${submit_concatenate}
submit_shuffle=$(qsub -W depend=afterok:${submit_concatenate} -l nodes=1:ppn=4:sl32g,walltime=01:01:00 submit_shuffle_h5_tt.sh)
echo ${submit_shuffle}
