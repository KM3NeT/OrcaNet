# -*- coding: utf-8 -*-
import csv
import os
__author__ = 'Michael Moser'

# This file converts simulated ORCA neutrino event .root files to .hdf5 files using the KM3Pipe (tohdf5 utility).
# Needs a .list file that contains the filepaths of all files that should be converted
# find /sps/km3net/users/kmcprod/JTE_NEMOWATER/muon-CC/3-100GeV -name "*.root" | sort > FilelistRoot.list

fp_list = raw_input('Please specify the filepath of the .list file which contains all filepaths of the to be converted files, e.g. /sps/km3net/users/mmoser/Listname.list: ')
print 'You entered (input) ', fp_list

# fp_hdf5_output = raw_input('Please specify the path of the hdf5 output files. e.g. /sps/km3net/users/mmoser/Data')
# print 'You entered (output) ', fp_hdf5_output

with open(fp_list, 'rt') as f:
    reader = csv.reader(f, delimiter='\t', skipinitialspace=False)
    filepaths_root = []

    for line in reader:
        filepaths_root.append(line[0])

# print filepaths_root
dir_inputlist = os.path.dirname(fp_list)

RootToHdf5_bashfile = open(dir_inputlist + '/RootToHdf5.sh', 'w')
RootToHdf5_bashfile.write('#!/bin/bash\n')
RootToHdf5_bashfile.write('source /afs/in2p3.fr/home/m/mmoser/setenvAA.sh\n')

for filepath in filepaths_root:
    RootToHdf5_bashfile.write('tohdf5 -o ' + dir_inputlist + '/' + str(os.path.splitext(os.path.basename(filepath))[0]) + '.hdf5 ' +
    str(filepath) + '\n')

RootToHdf5_bashfile.close()

os.system('qsub -P P_km3net -V -l ct=24:00:00 -l vmem=2G -l fsize=8G -l sps=1 -o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/RootToHdf5.sh')
print 'Finished! Submitted a job to convert the .root files to .hdf5 files as follows:\n' \
      'qsub -P P_km3net -V -l ct=24:00:00 -l vmem=2G -l fsize=8G -l sps=1 -o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/RootToHdf5.sh\n' \
      'Please change the resources like ct if you have thousands of files (>24h)'

# tohdf5 -o /sps/km3net/users/mmoser/Data/ORCA_JTE_NEMOWATER/hdf5/muon-CC/3-100GeV/test1 JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.1.root
# qsub -P P_km3net -V -l ct=24:00:00 -l vmem=2G -l fsize=8G -l sps=1 -o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser RootToHdf5.sh





