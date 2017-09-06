#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder"""

import numpy as np
import h5py
import os


# generator that returns arrays of batchsize events
# from hdf5 file
def generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, zero_center=False):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated. Ideally same as the chunksize in the h5 file.
    :param tuple n_bins: Number of bins for each dimension (x,y,z) in the h5 file.
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param bool zero_center: Specifies, if the input data (xs) should be zero-centered.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    dimensions = get_dimensions_encoding(batchsize, n_bins)

    #xs = np.array(np.zeros(dimensions)) # TODO redundant or better for performance?
    #ys = np.array(np.zeros((batchsize, class_type[0])))
    xs = np.zeros(dimensions, dtype=np.float32) # TODO redundant or better for performance?
    ys = np.zeros((batchsize, class_type[0]), dtype=np.float32)

    while 1:
        f = h5py.File(filepath, "r")
        filesize = len(f['y'])
        print "filesize = ", filesize

        if zero_center is True: xs_mean = get_mean_image(f, dimensions, filepath) #TODO if testing, load data generated from training sample!

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (filesize - batchsize):
        #while n_entries < batchsize*5:
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries : n_entries + batchsize]
            xs = np.reshape(xs, dimensions).astype('float32')

            #import sys
            #print xs[0][1]
            #print xs.shape # (32, 11, 18, 50, 1)
            #np.set_printoptions(threshold=10)
            #print np.amax(xs_mean)
            #print xs_mean
            #print xs_mean.shape # (11, 18, 50)

            if zero_center is True: xs = np.subtract(xs, xs_mean)
            # and mc info (labels)
            y_values = f['y'][n_entries:n_entries+batchsize]
            y_values = np.reshape(y_values, (batchsize, y_values.shape[1]))
            # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
            c = 0
            # TODO could be vectorized if performance is a bottleneck. Or just use dataflow from tensorpack!
            for y_val in y_values:
                ys[c] = encode_targets(y_val, class_type)
                c += 1

            # we have read one more batch from this file
            n_entries += batchsize
            #np.set_printoptions(threshold=np.inf)
            #print 'ys', ys.shape
            #print 'xs', xs.shape
            #print xs[0][1]
            #sys.exit()
            yield (xs, ys)
        f.close()


def get_dimensions_encoding(batchsize, n_bins):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in generate_batches_from_hdf5_file().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    if n_bins_x == 1:
        if n_bins_y == 1:
            print 'Using 2D projected data without dimensions x and y'
            dimensions = (batchsize, n_bins_z, n_bins_t, 1)
        elif n_bins_z == 1:
            print 'Using 2D projected data without dimensions x and z'
            dimensions = (batchsize, n_bins_y, n_bins_t, 1)
        elif n_bins_t == 1:
            print 'Using 2D projected data without dimensions x and t'
            dimensions = (batchsize, n_bins_y, n_bins_z, 1)
        else:
            print 'Using 3D projected data without dimension x'
            dimensions = (batchsize, n_bins_y, n_bins_z, n_bins_t, 1)

    elif n_bins_y == 1:
        if n_bins_z == 1:
            print 'Using 2D projected data without dimensions y and z'
            dimensions = (batchsize, n_bins_x, n_bins_t, 1)
        elif n_bins_t == 1:
            print 'Using 2D projected data without dimensions y and t'
            dimensions = (batchsize, n_bins_x, n_bins_z, 1)
        else:
            print 'Using 3D projected data without dimension y'
            dimensions = (batchsize, n_bins_x, n_bins_z, n_bins_t, 1)

    elif n_bins_z == 1:
        if n_bins_t == 1:
            print 'Using 2D projected data without dimensions z and t'
            dimensions = (batchsize, n_bins_x, n_bins_y, 1)
        else:
            print 'Using 3D projected data without dimension z'
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_t, 1)

    elif n_bins_t == 1:
        print 'Using 3D projected data without dimension t'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)

    else:
        print 'Using full 4D data'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    return dimensions


def get_mean_image(f, dimensions, filepath):
    """
    Returns the mean_image of a xs dataset by loading or calculating it.
    :param h5py.File f: h5py file object that contains the x dataset.
    :param tuple dimensions: dimensions tuple for 2D, 3D or 4D data.
    :param filepath: filepath of the input data, used as a str for saving the xs_mean_image.
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """

    if os.path.isfile(filepath) is True:
        print 'Loading an existing xs_mean_array in order to zero_center the data!'
        xs_mean = np.load(filepath + '_zero_center_mean.npy')

    else:
        print 'Calculating the xs_mean_array in order to zero_center the data! Warning: Memory must be as large as the inputfile!'
        # maybe astype np.float64 for increased precision
        xs_mean = np.mean(f['x'], axis=0)
        xs_mean = np.reshape(xs_mean, dimensions[1:])
        #xs_std = np.std(f['x'], axis=0, dtype=np.float64)
        np.save(filepath + '_zero_center_mean.npy', xs_mean)

    return xs_mean


def encode_targets(y_val, class_type):
    """
    Encodes the labels (classes) of the images.
    :param ndarray(ndim=1) y_val: Array that contains ALL event class information for one event.
           ---------------------------------------------------------------------------------------------------------------------------
           Current content: [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7,
                            up/down -> 8, categorical particle_types -> 9/10/11/12 (9: elec_NC, 10: elec_CC, 11: muon_CC, 12: tau_CC)]
           ---------------------------------------------------------------------------------------------------------------------------
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :return: ndarray(ndim=1) train_y: Array that contains the encoded class label information of the input event.
    """
    def get_class_up_down(dir_z):
        """
        Converts the zenith information (dir_z) to a binary up/down value.
        :param float32 dir_z: z-direction of the event_track (which contains dir_z).
        :return: int up_down_class_value: binary up/down class value for the event_track.
        """
        # analyze the track info to determine the class number
        up_down_class_value = int(np.sign(dir_z))
        if up_down_class_value == -1:
            up_down_class_value = 0

        return up_down_class_value

    def convert_particle_class_to_categorical(particle_type, is_cc, num_classes=4):
        """
        Converts the possible particle types (elec/muon/tau , NC/CC) to a categorical type that can be used as tensorflow input y
        :param int particle_type: Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
        :param int is_cc: Specifies the interaction channel. 0 = NC, 1 = CC.
        :param int num_classes: Specifies the total number of classes that will be discriminated later on by the CNN. I.e. 2 = elec_NC, muon_CC.
        :return: ndarray(ndim=1) categorical: returns the categorical event type. I.e. (particle_type=14, is_cc=1) -> [0,0,1,0] for num_classes=4.
        """
        if num_classes == 4:
            particle_type_dict = {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}  # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
        else:
            raise ValueError('A number of classes !=4 is currently not supported!')

        category = int(particle_type_dict[(abs(particle_type), is_cc)])
        categorical = np.zeros(num_classes, dtype='int')
        categorical[category] = 1
        return categorical


    if class_type == (2, 'muon-CC_to_elec-NC'):
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(2, dtype='float32')
        train_y[0] = categorical_type[0]
        train_y[1] = categorical_type[2]

    elif class_type == (1, 'muon-CC_to_elec-NC'): # only one neuron at the end of the cnn instead of two
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(1, dtype='float32')
        if categorical_type[0]!=0:
            train_y[0] = categorical_type[0]

    elif class_type == (2, 'muon-CC_to_elec-CC'):
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(2, dtype='float32')
        train_y[0] = categorical_type[1]
        train_y[1] = categorical_type[2]

        #print '------------------'
        #print y_val
        #print categorical_type
        #print train_y
        #print '------------------'

    elif class_type == (1, 'muon-CC_to_elec-CC'): # only one neuron at the end of the cnn instead of two
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(1, dtype='float32')
        if categorical_type[1]!=0:
            train_y[0] = categorical_type[1]

        #print '------------------'
        #print y_val
        #print categorical_type
        #print train_y
        #print '------------------'

    elif class_type == (2, 'up_down'): # up down, one neuron at the cnn end
        up_down_class = get_class_up_down(y_val[7]) # returns 0 or 1
        train_y = np.zeros(2, dtype='float32')
        train_y[up_down_class] = 1

    elif class_type == (1, 'up_down'): # up down, one neuron at the cnn end
        up_down_class = get_class_up_down(y_val[7]) # returns 0 or 1
        train_y = np.zeros(1, dtype='float32')
        train_y[0] = up_down_class

    else:
        print "Class type " + str(class_type) + " not supported!"
        return y_val

    return train_y


#unfinished
def predictAndPrintSome(model, testFile, printSize, numx, numy, numz, numt, nTargets):
    mySmallR = batchReaderHdf5()

    for j in range(printSize):
        x, yTrue = mySmallR.read_batch_from_file(testFile, 1, numx, numy, numz, numt, nTargets)

        # reconstruct values so we can look at them
        predictions = model.predict_on_batch(x)
        consideredNClasses = predictions.shape[1]
        yPred = predictions[0][0:consideredNClasses]
        if consideredNClasses > 1:
            yTrue = np.reshape(yTrue[0:consideredNClasses], consideredNClasses)
            yPred = np.reshape(yPred[0:consideredNClasses], consideredNClasses)

        # yTrue = decodeTargets(y, nTargets)
        # yPred = decodeTargets(predictions[0], nTargets)
        numberForPrint = min(consideredNClasses, 12)
        np.set_printoptions(precision=2)
        print "True and predicted:"
        print yTrue[0:numberForPrint]
        print yPred[0:numberForPrint]

#unfinished
def doTheEvaluation(model, number_of_classes, testFile, testSize, printSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, batchSize):
    if printSize > 0:
        # doesn't give any kind of accuracy
        predictAndPrintSome(model, testFile, printSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, number_of_classes)

    accuracy = 0.0
    mean_error = 0.0
    mean_squ_error = 0.0
    energy_diffs = []
    energy_diffs_rel = []
    angle_diffs = []
    angle_diffs_he = []  # greater than 10 TeV = 10^5 GeV = 0.5 in y_ETrue = y[3]

    myR = batchReaderHdf5()


    for l in range(int(testSize / batchSize + 0.5)):
        # reconstruct some values so we can look at them
        xs, ys = myR.read_batch_from_file(testFile, batchSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, number_of_classes)
        predictions = model.predict_on_batch(xs)
        consideredNClasses = predictions.shape[1]

        for j in range(predictions.shape[0]):
            yTrue = ys[j]
            yPred = predictions[j]
            if consideredNClasses > 1:
                # print yTrue
                yTrue = np.reshape(yTrue[0:consideredNClasses], consideredNClasses)
                yPred = np.reshape(yPred[0:consideredNClasses], consideredNClasses)

            if consideredNClasses == 1:
                diff = abs(yTrue - yPred)
                if diff < 0.1:
                    accuracy += 1.0
                mean_error += diff
                mean_squ_error += diff * diff

            if consideredNClasses > 1:
                for k in range(consideredNClasses):
                    diff = abs(yTrue[k] - yPred[k])
                    if diff < 0.1:
                        accuracy += 1.0
                    mean_error += diff
                    mean_squ_error += diff * diff

            if consideredNClasses >= 4:
                energy_diffs.append(abs(yTrue[3] - yPred[3]) / yTrue[3])
                energy_diffs_rel.append(np.log10((10.0 ** (10.0 * yPred[3])) / (10.0 ** (10.0 * yTrue[3]))))
                angle_diffs.append(angleDiff(yTrue[0:3], yPred[0:3]))
                if yTrue[3] > 0.5:  # if energy above 10TeV
                    angle_diffs_he.append(angleDiff(yTrue[0:3], yPred[0:3]))

    results = []
    nEventsTested = len(energy_diffs)
    nLabelsTested = nEventsTested * consideredNClasses
    results.append(["Number tested", nEventsTested])

    accuracy /= nLabelsTested
    results.append(["Accuracy", accuracy])
    mean_error /= nLabelsTested
    results.append(["Mean error", mean_error])
    mean_squ_error /= nLabelsTested
    results.append(["Mean squared error", mean_squ_error])
    mean_angle_diff = sum(angle_diffs) / nEventsTested
    results.append(["Mean angular error", mean_angle_diff])
    angle_diffs.sort()
    median_angle_diff = angle_diffs[int(0.5 * nEventsTested + 0.5)]
    results.append(["Median angular error", median_angle_diff])
    print ["Median angular error", median_angle_diff]
    angle_quantile_16 = angle_diffs[int(0.16 * nEventsTested + 0.5)]
    results.append(["Lower 16 angular error", angle_quantile_16])
    angle_quantile_84 = angle_diffs[int(0.84 * nEventsTested + 0.5)]
    results.append(["Upper 84 angular error", angle_quantile_84])
    angle_quantile_02 = angle_diffs[int(0.02 * nEventsTested + 0.5)]
    results.append(["Lower 02 angular error", angle_quantile_02])
    angle_quantile_98 = angle_diffs[int(0.98 * nEventsTested + 0.5)]
    results.append(["Upper 98 angular error", angle_quantile_98])

    mean_energy_diff = sum(energy_diffs) / nEventsTested
    results.append(["Mean error energy", mean_energy_diff])
    energy_diffs.sort()
    median_energy_diff = energy_diffs[int(0.5 * nEventsTested + 0.5)]
    results.append(["Median error energy", median_energy_diff])
    energy_quantile_16 = energy_diffs[int(0.16 * nEventsTested + 0.5)]
    results.append(["Lower 16 error energy", energy_quantile_16])
    energy_quantile_84 = energy_diffs[int(0.84 * nEventsTested + 0.5)]
    results.append(["Upper 84 error energy", energy_quantile_84])
    energy_quantile_02 = energy_diffs[int(0.02 * nEventsTested + 0.5)]
    results.append(["Lower 02 error energy", energy_quantile_02])
    energy_quantile_98 = energy_diffs[int(0.98 * nEventsTested + 0.5)]
    results.append(["Upper 98 error energy", energy_quantile_98])

    mean_energy_diff_rel = sum(energy_diffs_rel) / nEventsTested
    results.append(["Mean error energy relative", mean_energy_diff_rel])
    energy_diffs_rel.sort()
    median_energy_diff_rel = energy_diffs_rel[int(0.5 * nEventsTested + 0.5)]
    results.append(["Median error energy relative", median_energy_diff_rel])
    energy_quantile_rel_16 = energy_diffs_rel[int(0.16 * nEventsTested + 0.5)]
    results.append(["Lower 16 error energy relative", energy_quantile_rel_16])
    energy_quantile_rel_84 = energy_diffs_rel[int(0.84 * nEventsTested + 0.5)]
    results.append(["Upper 84 error energy relative", energy_quantile_rel_84])
    energy_quantile_rel_02 = energy_diffs_rel[int(0.02 * nEventsTested + 0.5)]
    results.append(["Lower 02 error energy relative", energy_quantile_rel_02])
    energy_quantile_rel_98 = energy_diffs_rel[int(0.98 * nEventsTested + 0.5)]
    results.append(["Upper 98 error energy relative", energy_quantile_rel_98])

    variance = 0.0
    for energy_diff in energy_diffs_rel:
        variance += (mean_energy_diff_rel - energy_diff) * (mean_energy_diff_rel - energy_diff)
    variance /= nEventsTested
    results.append(["Variance energy relative", variance])
    results.append(["Sigma energy relative", math.sqrt(variance)])

    mean_angle_diff_he = 0.0
    median_angle_diff_he = 0.0
    nEventsHE = len(angle_diffs_he)
    results.append(["Number tested highE", nEventsHE])
    if (nEventsHE > 0):
        mean_angle_diff_he = sum(angle_diffs_he) / nEventsHE
        results.append(["Mean error angular highE", mean_angle_diff_he])
        angle_diffs_he.sort()
        median_angle_diff_he = angle_diffs_he[int(0.5 * nEventsHE + 0.5)]
        results.append(["Median error angular highE", median_angle_diff_he])

    print ["Mean error", mean_error]
    return results