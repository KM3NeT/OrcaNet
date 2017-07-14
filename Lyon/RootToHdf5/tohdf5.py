# -*- coding: utf-8 -*-
import csv
import os

__author__ = 'Michael Moser'

# This file converts simulated ORCA neutrino event .root files to .hdf5 files using the KM3Pipe (tohdf5 utility).
# Needs a .list file that contains the filepaths of all files that should be converted
# find /sps/km3net/users/kmcprod/JTE_NEMOWATER/withMX/muon-CC/3-100GeV -name "*.root" | sort > FilelistRoot.list

fp_list = raw_input('Please specify the filepath of the .list file which contains all filepaths of the to be converted files,'
                    ' e.g. /sps/km3net/users/mmoser/Listname.list: ')
print 'You entered (input) ', fp_list

with open(fp_list, 'rt') as f:
    reader = csv.reader(f, delimiter='\t', skipinitialspace=False)
    filepaths_root = []

    for line in reader:
        filepaths_root.append(line[0])

dir_inputlist = os.path.dirname(fp_list)

# Write shell script: tohdf5
RootToHdf5_bashfile = open(dir_inputlist + '/RootToHdf5.sh', 'w')
RootToHdf5_bashfile.write('#!/bin/bash\n')
RootToHdf5_bashfile.write('source /afs/in2p3.fr/home/m/mmoser/.bash_profile\n')
RootToHdf5_bashfile.write('source /sps/km3net/users/mmoser/pyenv_km3pipe_7.2.2.sh\n')
RootToHdf5_bashfile.write('source /afs/in2p3.fr/home/m/mmoser/setenvAA_jpp8.sh\n')

for filepath in filepaths_root:
    RootToHdf5_bashfile.write('tohdf5 -o ' + dir_inputlist + '/' + str(os.path.splitext(os.path.basename(filepath))[0]) + '.h5 ' + str(filepath) + '\n')

RootToHdf5_bashfile.close()

os.system('qsub -P P_km3net -V -l ct=36:00:00 -l vmem=4G -l s_rss=2G -l fsize=1G -l sps=1 -o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/RootToHdf5.sh')
print '-----------------------------------------------------------------------------------------------------------------------------------------------------------'
print 'Finished! Submitted a job to convert the .root files to .hdf5 files as follows:\n' \
      'qsub -P P_km3net -V -l ct=36:00:00 -l vmem=2G -l fsize=8G -l sps=1 -o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/RootToHdf5.sh\n' \
      'Please change the resources like ct if you have thousands of files (>36h)'
print '-----------------------------------------------------------------------------------------------------------------------------------------------------------'

