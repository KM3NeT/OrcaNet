This repository contains the pipeline for the Deep Learning background suppression project within the context of the KM3NeT Tau appearance research.

The rough working plan is as follows:
1) Convert simulated ORCA neutrinos (muon/elec/tau) in .root files to .hdf5 files
2) Create projections (images) of the detector signal as an input for the DL network
3) Create the DL networks and train & evaluate their performance

The repository is structured in two parts, code for the CCIN2P3 in Lyon and the HPC.
