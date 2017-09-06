## Generating DL images based on KM3NeT-ORCA neutrino simulation data 

This directory contains code that produces 2D/3D/4D histograms ('images') for CNNs based on raw MC h5 files.

The main code for generating the images is located in 'h5_data_to_h5_input.py'. <br>
If the simulated .h5 files are not calibrated yet, you need to specify the directory of a .detx file in 'h5_data_to_h5_input.py'.

Currently, a bin size of 11x13x18x50 (x/y/z/t) is used for the final ORCA detector layout. 
