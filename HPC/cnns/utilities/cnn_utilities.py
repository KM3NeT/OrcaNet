#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder"""

import numpy as np
import h5py



# generator that returns arrays of batchsize events
# from hdf5 file
def generate_batches_from_hdf5_file(filename, batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, number_of_classes):

    dimensions = get_dimensions_encoding(batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    xs = np.array(np.zeros(dimensions))
    #xs = np.array( np.zeros(batchsize*n_bins_x*n_bins_y*n_bins_z*n_bins_t), ndmin=5 )
    #xs = np.reshape( xs, dimensions )
    ys = np.array(np.zeros((batchsize, number_of_classes)))
    #ys = np.array( np.zeros(batchsize*number_of_classes), ndmin=2 )
    #ys = np.reshape( ys, (batchsize, number_of_classes) )
    while 1:
        # Open the file
        f = h5py.File(filename, "r")
        # Check how many entries there are
        filesize = len(f['y'])
        print "filesize = ", filesize
        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (filesize-batchsize):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries:n_entries+batchsize]
            xs = np.reshape(xs, dimensions).astype(float)

            # and mc info (labels)
            y_values = f['y'][n_entries:n_entries+batchsize]
            y_values = np.reshape(y_values, (batchsize, y_values.shape[1]))
            # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
            c = 0
            for yVal in y_values:
                ys[c] = encodeTargets (yVal, number_of_classes)
                c += 1

            # we have read one batch more from this file
            n_entries += batchsize
            #np.set_printoptions(threshold=np.inf)
            #print ys
            #print xs
            yield (xs, ys)
        f.close()


# returns the dimensions tuple for 2,3 and 4 dimensional data
# we don't have to write separate functions with different argument lists for different dimensions but can always use numx, numy, numz, numt
def get_dimensions_encoding(batchsize, numx, numy, numz, numt):
    dimensions = (batchsize,numx,numy,numz,numt)
    if numx == 1:
        if numy == 1:
            print "2D case without dimensions x and y"
            dimensions = (batchsize,numz,numt,1)
        elif numz == 1:
            print "2D case without dimensions x and z"
            dimensions = (batchsize,numy,numt,1)
        elif numt == 1:
            print "2D case without dimensions x and t"
            dimensions = (batchsize,numy,numz,1)
        else:
            # print "3D case without dimension x"
            dimensions = (batchsize,numy,numz,numt,1)

    elif numy == 1:
        if numz == 1:
            print "2D case without dimensions y and z"
            dimensions = (batchsize,numx,numt,1)
        elif numt == 1:
            print "2D case without dimensions y and t"
            dimensions = (batchsize,numx,numz,1)
        else:
            print "3D case without dimension y"
            dimensions = (batchsize,numx,numz,numt,1)

    elif numz == 1:
        if numt == 1:
            print "2D case without dimensions z and t"
            dimensions = (batchsize,numx,numy,1)
        else:
            print "3D case without dimension z"
            dimensions = (batchsize,numx,numy,numt,1)

    elif numt == 1:
        print "3D case without dimension t"
        dimensions = (batchsize,numx,numy,numz,1)

    else:	# 4 dimensional
        # print "4D case"
        dimensions = (batchsize,numx,numy,numz,numt)

    return dimensions


def encodeTargets(mcinfo, number_of_classes):
    if number_of_classes == 16:
        # everything at once:
        temp = mcinfo
        temp[5] = np.log10(mcinfo[5]) / 10.0

        temp[2] = 0.5 * (mcinfo[2] + 1.0)
        temp[3] = 0.5 * (mcinfo[3] + 1.0)
        temp[4] = 0.5 * (mcinfo[4] + 1.0)

        numPids = 9
        pids = np.zeros(numPids)

        pid = mcinfo[1]
        # just hardcode the mapping
        if pid == -12:  # a nu e
            pids[1] = 1.0
        elif pid == 12:  # nu e
            pids[2] = 1.0
        elif pid == -14:  # a nu mu
            pids[3] = 1.0
        elif pid == 14:  # nu mu
            pids[4] = 1.0
        elif pid == -16:  # a nu tau
            pids[5] = 1.0
        elif pid == 16:  # nu tau
            pids[6] = 1.0
        elif pid == -13:  # a mu
            pids[7] = 1.0
        elif pid == 13:  # mu
            pids[8] = 1.0
        else:  # if it's nothing else we know: we don't know what it is ;-)
            pids[0] = 1.0
        # TODO: Probably pid and isCC work better if there are classes e.g. for numuCC and numuNC and nueCC and nueNC
        # instead of single classes for numu and nue but a combined flag is_CC_or_NC for all flavour
        # especially for numuCC and numuNC

        trainY = np.concatenate([np.reshape(temp[2:9], len(temp[2:9]), 1), np.reshape(pids, numPids, 1)])
        # 0 1 2 3 4  5  6  7          8    9   10    11   12     13    14  15
        # x y z E cc by ud unknownPid anue nue anumu numu anutau nutau amu mu
        return trainY

    elif number_of_classes == 6:
        # direction, energy and iscc and bjorken-y:
        temp = mcinfo
        # energy
        temp[5] = np.log10(mcinfo[5]) / 10.0
        # direction
        temp[2] = 0.5 * (mcinfo[2] + 1.0)
        temp[3] = 0.5 * (mcinfo[3] + 1.0)
        temp[4] = 0.5 * (mcinfo[4] + 1.0)

        trainY = np.reshape(temp[2:8], number_of_classes, 1)
        # 0 1 2 3 4  5
        # x y z E cc by
        return trainY

    elif number_of_classes == 4:
        # direction and energy:
        temp = mcinfo
        temp[5] = np.log10(mcinfo[5]) / 10.0

        temp[2] = 0.5 * (mcinfo[2] + 1.0)
        temp[3] = 0.5 * (mcinfo[3] + 1.0)
        temp[4] = 0.5 * (mcinfo[4] + 1.0)

        trainY = np.reshape(temp[2:6], number_of_classes, 1)
        # 0 1 2 3
        # x y z E
        return trainY

    elif number_of_classes == 1:
        # energy:
        temp = mcinfo
        temp[5] = np.log10(mcinfo[5]) / 10.0
        return np.reshape(temp[5:6], number_of_classes, 1)

    else:
        print "Number of targets (" + str(number_of_classes) + ") not supported!"
        return mcinfo