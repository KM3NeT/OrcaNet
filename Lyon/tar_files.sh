#!/usr/bin/env bash

# Michael: submit with
# qsub -P P_km3net -V -l ct=48:00:00 -l vmem=8G -l s_rss=4G -l fsize=10G -l sps=1 -l irods=1 -o /sps/km3net/users/mmoser -e /sps/km3net/users/mmoser tar_files.sh

tar_files()
{
dir_files_full=${dir_files}/${p_type}/${energy}
cd ${dir_files_full}

for i in {12..25}
do

index_first_file=$((i*50 + 1))
index_last_file=$((index_first_file + 49))

tar_filename=${p_type}_${energy}_${index_first_file}-${index_last_file}${f_ending}.tar.gz

eval tar -czvf ${tar_filename} ${basename}.{${index_first_file}..${index_last_file}}${f_ending}
iput ${tar_filename} ${irods_put_folder}

done
}

### JTE

# muon-CC: for i in {12..35}
# elec-CC: for i in {12..25}

# muon-CC, 3-100GeV
#dir_files="/sps/km3net/users/mmoser/Sim_Productions/gseagen_km3sim_prod/sim_files/muon-CC/3-100GeV/triggered/withMX"
#p_type="muon-CC"
#energy="3-100GeV"
#f_ending=".root"
#basename="JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016"
#irods_put_folder="/in2p3/km3net/mc/atm_neutrino/KM3NeT_ORCA_115_23m_9m/v1.1.1/JTE"

# elec-CC, 3-100GeV
# TODO change range of i
dir_files="/sps/km3net/users/mmoser/Sim_Productions/gseagen_km3sim_prod/sim_files/withMX"
p_type="elec-CC"
energy="3-100GeV"
f_ending=".root"
basename="JTE.KM3Sim.gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016"
irods_put_folder="/in2p3/km3net/mc/atm_neutrino/KM3NeT_ORCA_115_23m_9m/v1.1.1/JTE"
tar_files

### gseagen

# muon-CC, 3-100GeV
#dir_files="/sps/km3net/users/mmoser/Sim_Productions/gseagen_km3sim_prod/sim_files"
#p_type="muon-CC"
#energy="3-100GeV"
#f_ending=".evt"
#basename="gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016"
#irods_put_folder="/in2p3/km3net/mc/atm_neutrino/KM3NeT_ORCA_115_23m_9m/v1.0/gSeaGen"

# elec-CC, 3-100GeV
# TODO change range of i
dir_files="/sps/km3net/users/mmoser/Sim_Productions/gseagen_km3sim_prod/sim_files"
p_type="elec-CC"
energy="3-100GeV"
f_ending=".evt"
basename="gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016"
irods_put_folder="/in2p3/km3net/mc/atm_neutrino/KM3NeT_ORCA_115_23m_9m/v1.0/gSeaGen"

tar_files

f_ending=".root"
tar_files

### KM3Sim

## muon-CC, 3-100GeV
#dir_files="/sps/km3net/users/mmoser/Sim_Productions/gseagen_km3sim_prod/sim_files"
#p_type="muon-CC"
#energy="3-100GeV"
#f_ending=".evt"
#basename="KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016"
#irods_put_folder="/in2p3/km3net/mc/atm_neutrino/KM3NeT_ORCA_115_23m_9m/v1.1.0/KM3Sim"
#tar_files

## elec-CC, 3-100GeV
# TODO change range of i
dir_files="/sps/km3net/users/mmoser/Sim_Productions/gseagen_km3sim_prod/sim_files"
p_type="elec-CC"
energy="3-100GeV"
f_ending=".evt"
basename="KM3Sim.gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016"
irods_put_folder="/in2p3/km3net/mc/atm_neutrino/KM3NeT_ORCA_115_23m_9m/v1.1.0/KM3Sim"
tar_files

