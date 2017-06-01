#!/bin/bash
#
#PBS -l nodes=1:ppn=4:s132,walltime=06:00:00
#
# first non-empty non-comment line ends PBS options
cd  ${PBS_O_WORKDIR}

n=${PBS_ARRAYID}
i=$((1+((${n}-1) * 4)))

thread1=${i}
thread2=$((${i} + 1))
thread3=$((${i} + 2))
thread4=$((${i} + 3))

CodeFolder=/home/woody/capn/mppi033h/Code/HPC/h5ToHisto
HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5/muon-CC/3-100GeV

ParticleType=muon-CC
FileName=JTE.KM3Sim.gseagen.${ParticleType}.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016

# run
(taskset -c 0  time python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread1}.hdf5 > ${FileName}.${thread1}.txt) &
(taskset -c 1  time python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread2}.hdf5 > ${FileName}.${thread2}.txt) &
(taskset -c 2  time python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread3}.hdf5 > ${FileName}.${thread3}.txt) &
(taskset -c 3  time python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread4}.hdf5 > ${FileName}.${thread4}.txt) &

# wait for all background processes to finish
wait

# not used: PBS -o /home/woody/capn/mppi033h/logs/submit_h5_to_histo.out -e /home/woody/capn/mppi033h/logs/submit_h5_to_histo.err






