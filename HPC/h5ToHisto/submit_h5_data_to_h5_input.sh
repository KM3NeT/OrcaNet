#!/bin/bash
#
#PBS -l nodes=1:ppn=4,walltime=01:30:00
#PBS -o /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.out -e /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.err
# first non-empty non-comment line ends PBS options

# Submit with 'qsub -t 1-10 submit_h5_data_to_h5_input.sh' # fix km3pipe six before doing this
# This script uses the h5_data_to_h5_input.py file in order to convert all 600 (muon/elec/tau) .h5 raw files to .h5 2D/3D projection files (CNN input).
# The total amount of simulated files for each event type in ORCA is 600 -> file 1-600
# The files should be converted in batches of files_per_job=60 files per job

# load env
source activate /home/hpc/capn/mppi033h/.virtualenv/env_km3pipe_tb/

n=${PBS_ARRAYID}
i=$((1+((${n}-1) * 4)))

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/h5ToHisto
cd ${CodeFolder}

#ParticleType=muon-CC #muon-CC
#ParticleType=elec-NC #elec-NC
ParticleType=elec-CC #elec-CC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016 #muon-CC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-3.4E6-1bin-3.0gspec.ORCA115_9m_2016 #elec-NC
FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016 #elec-CC

#with run_id
HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/h5/calibrated/with_run_id/without_mc_time_fix/${ParticleType}/3-100GeV

# run

files_per_job=60 # total number of files per job
no_of_loops=$((${files_per_job}/4)) # divide by 4 cores -> e.g, 15 4-core loops needed for files_per_job=60
file_no_start=$((1+((${n}-1) * ${files_per_job}))) # filenumber of the first file that is being processed by this script (depends on JobArray variable 'n')

for (( k=1; k<=${no_of_loops}; k++ ))
do
    file_no_loop_start=$((${file_no_start}+(k-1)*4))
    thread1=${file_no_loop_start}
    thread2=$((${file_no_loop_start} + 1))
    thread3=$((${file_no_loop_start} + 2))
    thread4=$((${file_no_loop_start} + 3))

    (time taskset -c 0  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread1}.h5 > ./logs/cout/${FileName}.${thread1}.txt) &
    (time taskset -c 1  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread2}.h5 > ./logs/cout/${FileName}.${thread2}.txt) &
    (time taskset -c 2  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread3}.h5 > ./logs/cout/${FileName}.${thread3}.txt) &
    (time taskset -c 3  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread4}.h5 > ./logs/cout/${FileName}.${thread4}.txt) &
    wait
done

