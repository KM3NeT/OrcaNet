#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code for running CNN's."""

import keras as ks

from models.cnn_models import *
from utilities.cnn_utilities import *


def execute_cnn(n_bins):

    number_of_classes = 2

    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    testfile, testsize = 'input/numuyztShufTail54921.csv.h5', 5000
    trainfiles, trainsize = ['input/numuyztShufHead270k.csv.h5'], 270000

    batchsize = 32
    print "Batchsize = ", batchsize

    modelname = "model_3d_xyz_numuCC_vs_nueNC_epoch"
    restart_index = 0  # 4 targets, 6xhdf5, bs 32, mse, adam

    if restart_index == 0:
        model = define_3d_model_xyz(number_of_classes, [n_bins_x, n_bins_y, n_bins_z])
    else:
        model = ks.models.load_model("models/trained" + modelname + str(restart_index) + ".h5")

    model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=["mean_squared_error"])
    # model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
    # model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mean_absolute_error"])
    model.summary()

    printSize = 5
    i = restart_index

    trainsize = 100000

    while 1:
        # process all hdf5 files, full epoch
        for f in trainfiles:
            i += 1
            print "Training ", i, " on file ", f
            model.fit_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, number_of_classes),
                                steps_per_epoch=int(trainsize / batchsize), epochs=1, verbose=1, max_q_size=1)
            # store the trained model
            model.save("models/" + modelname + str(i) + ".h5")
            if testfile != "":
                results = doTheEvaluation(model, number_of_classes, testfile, testsize, printSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, batchsize)


if __name__ == '__main__':
    execute_cnn(n_bins=[11,13,18,50])

