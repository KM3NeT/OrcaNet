#!/bin/bash
#
#PBS -l nodes=1:ppn=4:sl32g,walltime=5:00:00
#PBS -o /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.out -e /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.err
# first non-empty non-comment line ends PBS options

# Submit with 'qsub -t 1-10 submit_h5_data_to_h5_input.sh' # fix km3pipe six before doing this
# This script uses the h5_data_to_h5_input.py file in order to convert all 600 (muon/elec/tau) .h5 raw files to .h5 2D/3D projection files (CNN input).
# The total amount of simulated files for each event type in ORCA is 600 -> file 1-600
# The files should be converted in batches of files_per_job=60 files per job

# load env
source activate /home/hpc/capn/mppi033h/.virtualenv/env_km3pipe_tb/

n=${PBS_ARRAYID}

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/h5ToHisto
cd ${CodeFolder}

ParticleType=muon-CC #muon-CC
#ParticleType=elec-NC #elec-NC
#ParticleType=elec-CC #elec-CC
#ParticleType=tau-CC #tau-CC
# ----- 3-100GeV------
FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016 #muon-CC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-3.4E6-1bin-3.0gspec.ORCA115_9m_2016 #elec-NC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016 #elec-CC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.3.4-100GeV-2.0E8-1bin-3.0gspec.ORCA115_9m_2016 #tau-CC
HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/calibrated/without_mc_time_fix/3-100GeV/${ParticleType}
# ----- 3-100GeV------
# ----- 1-5GeV------
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.1-5GeV-9.2E5-1bin-1.0gspec.ORCA115_9m_2016 #muon-CC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.1-5GeV-2.2E6-1bin-1.0gspec.ORCA115_9m_2016 #elec-NC
#FileName=JTE.KM3Sim.gseagen.${ParticleType}.1-5GeV-2.7E5-1bin-1.0gspec.ORCA115_9m_2016 #elec-CC
#HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/calibrated/without_mc_time_fix/1-5GeV/${ParticleType}
# ----- 1-5GeV------

# tau-CC: 180 | 3-100GeV: muon-CC 240 , elec-NC 120, elec-CC 120
files_per_job=60 # total number of files per job, e.g. 10 jobs for 600: 600/10 = 60

# run
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

