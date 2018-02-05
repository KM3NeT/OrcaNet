## Deep Learning in KM3NeT-ORCA <br />
Find more information about KM3NeT on [www.km3net.org](http://www.km3net.org).

This repository contains contains code that is used for a Deep Learning project with the neutrino telescope KM3NeT, which records 4D data (xyz,t). <br />
The goal of the project is to classify the different event topologies & properties of KM3NeT events. <br />
These characteristics are then used in a research project to detect tau neutrinos with KM3NeT-ORCA. <br />


The repository is structured in code for two computing centers that I use: the CCIN2P3 in Lyon and the HPC in Erlangen. <br />
The Lyon folder just contains code to convert .root files to .h5 files, while the HPC folder contains the main code. <br />

In the HPC folder, two subprojects exist:
- h5ToHisto: Collects raw .h5 simulated detector data and generates 2D/3D/4D histograms ('images') that can be used for Deep Learning applications
- cnns: Code for running cnns based on the generated images from h5ToHisto. It currently contains
    - the main code for running a cnn ('run_cnn.py')
    - different model architectures e.g. ResNets
    - utility code: concatenating h5 files, shuffling h5 files, h5 cnn read-in, multi-gpu cnn support,...


Feel free to use tools like concatenate_h5.py or shuffle_h5.py (and the other stuff as well of course) for your personal usecase.



