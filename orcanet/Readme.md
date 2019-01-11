## Code for running a CNN <br />

This directory contains code for running a convolutional neural network based on the projections from h5ToHisto. The main script to run a CNN is 'run_cnn.py'. 

Other than that, the directory contains some utility code and models:
- /utilities/concatenate_h5.py for concatenating .h5 files (each dataset needs to have the same number of rows!)
- /utilities/shuffle_h5.py for shuffling the datasets of a .h5 file
- /utilities/multi-gpu for TF multi-gpu support
- /utilities/cnn_utilities for general stuff like reading the .h5 images
- models: Currently, simple VGG-like models and Wide-ResNets are available.
