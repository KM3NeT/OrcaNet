## Deep Learning in KM3NeT-ORCA <br />
Find more information about KM3NeT on [www.km3net.org](http://www.km3net.org).

This repository contains contains code that is used for a Deep Learning project with the neutrino telescope KM3NeT, which records 4D data (xyz,t). <br />
The goal of the project is to classify the different event topologies & properties of KM3NeT events. <br />
These characteristics are then used in a research project to detect tau neutrinos with KM3NeT-ORCA. <br />


The repository currently contains two folders: the CCIN2P3 in Lyon and the HPC in Erlangen. <br />
- cnns: Code for running cnns based on the generated images from OrcaSong. It currently contains
    - the main code for running a cnn ('run_cnn.py')
    - different model architectures e.g. ResNets
    - utility code: concatenating h5 files, shuffling h5 files, h5 cnn read-in, multi-gpu cnn support,...

- utilities: utility tools that are not related to running the cnn.



