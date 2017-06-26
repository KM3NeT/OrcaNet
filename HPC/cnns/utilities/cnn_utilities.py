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
    ys = np.array(np.zeros((batchsize, number_of_classes)))

    while 1:
        # Open the file
        f = h5py.File(filename, "r")
        # Check how many entries there are
        filesize = len(f['y'])
        print "filesize = ", filesize
        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (filesize - batchsize):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries : n_entries + batchsize]
            xs = np.reshape(xs, dimensions).astype(float) #float32?

            # and mc info (labels)
            y_values = f['y'][n_entries:n_entries+batchsize]
            y_values = np.reshape(y_values, (batchsize, y_values.shape[1]))
            # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
            c = 0
            for y_val in y_values:
                ys[c] = encode_targets(y_val, number_of_classes)
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
            print "3D case without dimension x"
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
        print "4D case"
        dimensions = (batchsize,numx,numy,numz,numt)

    return dimensions


def encode_targets(y_val, number_of_classes):

    if number_of_classes == 2:
        # [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7,
        #  up/down -> 8, categorical particle_types -> 9/10/11/12 (9: elec_NC, 10: elec_CC, 11: muon_CC, 12: tau_CC)]
        # particle type, only two types:muon-CC and elec-NC
        train_y = np.zeros(2, dtype='float32')
        train_y[0] = y_val[9]
        train_y[1] = y_val[11]

        return train_y

    # if number_of_classes == 16:
    #     # everything at once:
    #     temp = y_val
    #     temp[5] = np.log10(y_val[5]) / 10.0
    #
    #     temp[2] = 0.5 * (y_val[2] + 1.0)
    #     temp[3] = 0.5 * (y_val[3] + 1.0)
    #     temp[4] = 0.5 * (y_val[4] + 1.0)
    #
    #     numPids = 9
    #     pids = np.zeros(numPids)
    #
    #     pid = y_val[1]
    #     # just hardcode the mapping
    #     if pid == -12:  # a nu e
    #         pids[1] = 1.0
    #     elif pid == 12:  # nu e
    #         pids[2] = 1.0
    #     elif pid == -14:  # a nu mu
    #         pids[3] = 1.0
    #     elif pid == 14:  # nu mu
    #         pids[4] = 1.0
    #     elif pid == -16:  # a nu tau
    #         pids[5] = 1.0
    #     elif pid == 16:  # nu tau
    #         pids[6] = 1.0
    #     elif pid == -13:  # a mu
    #         pids[7] = 1.0
    #     elif pid == 13:  # mu
    #         pids[8] = 1.0
    #     else:  # if it's nothing else we know: we don't know what it is ;-)
    #         pids[0] = 1.0
    #     # TODO: Probably pid and isCC work better if there are classes e.g. for numuCC and numuNC and nueCC and nueNC
    #     # instead of single classes for numu and nue but a combined flag is_CC_or_NC for all flavour
    #     # especially for numuCC and numuNC
    #
    #     train_y = np.concatenate([np.reshape(temp[2:9], len(temp[2:9]), 1), np.reshape(pids, numPids, 1)])
    #     # 0 1 2 3 4  5  6  7          8    9   10    11   12     13    14  15
    #     # x y z E cc by ud unknownPid anue nue anumu numu anutau nutau amu mu
    #     return train_y
    #
    # elif number_of_classes == 6:
    #     # direction, energy and iscc and bjorken-y:
    #     temp = y_val
    #     # energy
    #     temp[5] = np.log10(y_val[5]) / 10.0
    #     # direction
    #     temp[2] = 0.5 * (y_val[2] + 1.0)
    #     temp[3] = 0.5 * (y_val[3] + 1.0)
    #     temp[4] = 0.5 * (y_val[4] + 1.0)
    #
    #     train_y = np.reshape(temp[2:8], number_of_classes, 1)
    #     # 0 1 2 3 4  5
    #     # x y z E cc by
    #     return train_y
    #
    # elif number_of_classes == 4:
    #     # direction and energy:
    #     temp = y_val
    #     temp[5] = np.log10(y_val[5]) / 10.0
    #
    #     temp[2] = 0.5 * (y_val[2] + 1.0)
    #     temp[3] = 0.5 * (y_val[3] + 1.0)
    #     temp[4] = 0.5 * (y_val[4] + 1.0)
    #
    #     train_y = np.reshape(temp[2:6], number_of_classes, 1)
    #     # 0 1 2 3
    #     # x y z E
    #     return train_y
    #
    # elif number_of_classes == 1:
    #     # energy:
    #     temp = y_val
    #     temp[5] = np.log10(y_val[5]) / 10.0
    #     return np.reshape(temp[5:6], number_of_classes, 1)

    else:
        print "Number of targets (" + str(number_of_classes) + ") not supported!"
        return y_val


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


def doTheEvaluation(model, number_of_classes, testFile, testSize, printSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, batchSize):
    if printSize > 0:
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