## OrcaNet: A framework for Deep Learning in KM3NeT <br />

[![alt text][image_1]][hyperlink_1] [![alt text][image_2]][hyperlink_2]

  [hyperlink_1]: https://git.km3net.de/OrcaNet/pipelines
  [image_1]: https://git.km3net.de/ml/OrcaNet/badges/master/build.svg

  [hyperlink_2]: https://ml.pages.km3net.de/OrcaNet
  [image_2]: https://examples.pages.km3net.de/km3badges/docs-latest-brightgreen.svg

Find more information about KM3NeT on [www.km3net.org](http://www.km3net.org).

This repository contains contains code that is used for a Deep Learning project with the neutrino telescope KM3NeT, which records 4D data (xyz,t).
The goal of the project is to classify the different event topologies & properties of KM3NeT events. <br />
These characteristics are then used in a research project to detect tau neutrinos with KM3NeT-ORCA. <br />

The repository currently contains two folders: <br />
- cnns: Code for running cnns based on the generated images from OrcaSong. It currently contains
    - the main code for running a cnn ('run_cnn.py')
    - different model architectures e.g. ResNets
    - utility code: concatenating h5 files, shuffling h5 files, h5 cnn read-in, multi-gpu cnn support,...

- utilities: utility tools that are not related to running the cnn.