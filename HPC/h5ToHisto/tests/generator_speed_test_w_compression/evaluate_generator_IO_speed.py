#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder"""

import numpy as np
import h5py
import timeit
import cProfile

def generate_batches_from_hdf5_file():

    #filepath = './JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yz_NO_COMPRESSION_NOT_CHUNKED.h5' #2D, no compression
    #filepath = './JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yz_LZF_CHUNKED.h5' #2D, LZF

    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_NO_COMPRESSION_NOT_CHUNKED.h5' #3D, no compression
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_NO_COMPRESSION_CHUNKED.h5' #3D, no compression, chunked
    filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_LARGE_NO_COMPRESSION_CHUNKED.h5' #3D, no compression, chunked, LARGE
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_LZF_CHUNKED.h5' #3D, LZF
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_LZF_LARGE.h5' #3D, LZF, LARGE, Shuffle True
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_LZF_SHUFFLETRUE.h5' #3D, LZF, Shuffle=True
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_GZIP_COMPROPT_1.h5' #3D, GZIP
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_456_yzt_GZIP_COMPROPT_1_CHUNKSIZE_16.h5' #3D, GZIP, chunksize=16

    #print 'Testing generator on file ' + filepath
    #batchsize = 16
    batchsize = 32

    #dimensions = (batchsize, 13, 18, 1) # 2D
    dimensions = (batchsize, 13, 18, 50, 1)  # 3D



    xs = np.zeros(dimensions, dtype=np.float32) # Redundant or better for performance? Answer: Both are better for performance
    ys = np.zeros((batchsize, 2), dtype=np.float32) #

    f = h5py.File(filepath, "r")
    filesize = len(f['y'])
    print filesize

    n_entries = 0

    while n_entries < (filesize - batchsize):
        xs = f['x'][n_entries : n_entries + batchsize]
        xs = np.reshape(xs, dimensions).astype(np.float32)

        y_values = f['y'][n_entries:n_entries+batchsize]
        y_values = np.reshape(y_values, (batchsize, y_values.shape[1]))
        #ys = np.zeros((batchsize, 2), dtype=np.float32)

        for c, y_val in enumerate(y_values):
            ys[c] = y_val[0:2] # just for testing

        n_entries += batchsize
        #print n_entries
        yield (xs, ys)
    f.close()


number = 100
#t = timeit.timeit(generate_batches_from_hdf5_file, number = number)
#t = timeit.Timer(stmt="list(generate_batches_from_hdf5_file())", setup="from __main__ import generate_batches_from_hdf5_file")
#print t.timeit(number) / number
#print str(number) + 'loops, on average ' + str(t.timeit(number) / number *1000) + 'ms'

pr = cProfile.Profile()
pr.enable()

t = timeit.Timer(stmt="list(generate_batches_from_hdf5_file())", setup="from __main__ import generate_batches_from_hdf5_file")
print str(number) + 'loops, on average ' + str(t.timeit(number) / number *1000) + 'ms'

pr.disable()

pr.print_stats(sort='time')

# TODO check with blosc