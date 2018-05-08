# -*- coding: utf-8 -*-
import csv
import os
import subprocess

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

# Write first shell script: tohdf5
RootToHdf5_bashfile = open(dir_inputlist + '/RootToHdf5.sh', 'w')
RootToHdf5_bashfile.write('#!/bin/bash\n')
RootToHdf5_bashfile.write('pip list\n')
#RootToHdf5_bashfile.write('source /afs/in2p3.fr/home/m/mmoser/.bash_profile\n')
#RootToHdf5_bashfile.write('pip list\n')
RootToHdf5_bashfile.write('source /sps/km3net/users/mmoser/pyenv_km3pipe.sh\n')
RootToHdf5_bashfile.write('pip list\n')
RootToHdf5_bashfile.write('source /sps/km3net/users/mmoser/setenvAA_jpp8_sl7.sh\n')
RootToHdf5_bashfile.write('pip list\n')

for filepath in filepaths_root:
    RootToHdf5_bashfile.write('tohdf5 --skip-header -o ' + dir_inputlist + '/' + str(os.path.splitext(os.path.basename(filepath))[0]) + '.h5 ' + str(filepath) + '\n')

RootToHdf5_bashfile.close()


#Write second shell script: calibrate
calibrate_bashfile = open(dir_inputlist + '/h5_calibrate_job.sh', 'w')
calibrate_bashfile.write('#!/bin/bash\n')
calibrate_bashfile.write('pip list\n')
calibrate_bashfile.write('source /sps/km3net/users/mmoser/pyenv_km3pipe.sh\n')
calibrate_bashfile.write('pip list\n')
calibrate_bashfile.write('source /sps/km3net/users/mmoser/setenvAA_jpp8_sl7.sh\n')
calibrate_bashfile.write('pip list\n')

detector_file = '/sps/km3net/users/mmoser/orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.detx' # standard ORCA
#detector_file = '/sps/km3net/users/mmoser/orca_115strings_av23min20mhorizontal_18OMs_15mvertical_v1.detx' # 15m ORCA
#detector_file = '/sps/km3net/users/mmoser/orca_115strings_av23min20mhorizontal_18OMs_12mvertical_v1.detx' # 12m ORCA
#detector_file = '/sps/km3net/users/mmoser/orca_115strings_av23min20mhorizontal_18OMs_9mvertical_v1.detx' # 9m ORCA
#detector_file = '/sps/km3net/users/mmoser/orca_115strings_av23min20mhorizontal_18OMs_6mvertical_v1.detx' # 6m ORCA
#detector_file = '/sps/km3net/users/mmoser/orca_115strings_av23min20mhorizontal_18OMs_4p5mvertical_v1.detx' # 4.5m ORCA

for filepath in filepaths_root:
    calibrate_bashfile.write('calibrate ' + detector_file + ' '
                              + dir_inputlist + '/' + str(os.path.splitext(os.path.basename(filepath))[0]) + '.h5' + '\n')

calibrate_bashfile.close()

#Write third shell script: Submit 1 and 2
submit_bashfile = open(dir_inputlist + '/submit_tohdf5_calibrate_job.sh', 'w')
submit_bashfile.write('#!/bin/bash\n')
submit_bashfile.write('TOHDF5=$(qsub -P P_km3net -V -l ct=48:00:00 -l vmem=4G -l s_rss=2G -l fsize=1G -l sps=1 '
                      '-o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/RootToHdf5.sh)\n')
submit_bashfile.write("TOHDF5=`echo $TOHDF5 | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`\n")
submit_bashfile.write('CALIBRATE=$(qsub -hold_jid $TOHDF5 -P P_km3net -V -l ct=48:00:00 -l vmem=4G -l s_rss=2G -l fsize=1G -l sps=1 '
                      '-o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/h5_calibrate_job.sh)\n')
submit_bashfile.write('echo $CALIBRATE\n')
submit_bashfile.write('exit 0')
submit_bashfile.close()


os.chmod(dir_inputlist + '/submit_tohdf5_calibrate_job.sh', 0744)
subprocess.Popen([dir_inputlist + '/submit_tohdf5_calibrate_job.sh'], shell = True)

print '-----------------------------------------------------------------------------------------------------------------------------------------------------------'
print 'Finished! Submitted a job to convert the .root files to .hdf5 files and to calibrate them as follows:\n' \
      'tohdf5: qsub -P P_km3net -V -l ct=48:00:00 -l vmem=4G -l s_rss=2G -l fsize=1G -l sps=1 ' \
      '-o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/RootToHdf5.sh\n' \
      'calibrate: qsub -hold_jid $TOHDF5 -P P_km3net -V -l ct=48:00:00 -l vmem=4G -l s_rss=2G -l fsize=1G -l sps=1 ' \
      '-o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser ' + dir_inputlist + '/h5_calibrate_job.sh\n ' \
      'Please change the resources like ct if you have thousands of files (>36h)'
print '-----------------------------------------------------------------------------------------------------------------------------------------------------------'

