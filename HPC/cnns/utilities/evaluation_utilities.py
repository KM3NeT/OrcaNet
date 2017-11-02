#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code for the evaluation of a network's performance after training."""

import h5py
import numpy as np
import keras as ks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .cnn_utilities import generate_batches_from_hdf5_file

#------------- Functions used in evaluating the performance of model -------------#

def make_performance_array_energy_correct(model, f, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, samples=None):
    """
    Creates an energy_correct array based on test_data that specifies for every event, if the model's prediction is True/False.
    :param ks.model.Model/Sequential model: Fully trained Keras model of a neural network.
    :param str f: Filepath of the file that is used for making predctions.
    :param tuple n_bins: The number of bins for each dimension (x,y,z,t) in the testfile.
    :param (int, str) class_type: The number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for predicting.
    :param ndarray xs_mean: mean_image of the x dataset if zero-centering is enabled.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped in the generator.
    :param None/int samples: Number of events that should be predicted. If samples=None, the whole file will be used.
    :return: ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    """
    # TODO only works for a single test_file till now
    generator = generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean, yield_mc_info=True, swap_col=swap_4d_channels) # f_size=samples prob not necessary

    if samples is None: samples = len(h5py.File(f, 'r')['y'])
    steps = samples/batchsize

    arr_energy_correct = None
    for s in xrange(steps):
        if s % 100 == 0:
            print 'Predicting in step ' + str(s)
        xs, y_true, mc_info = next(generator)
        y_pred = model.predict_on_batch(xs)

        # check if the predictions were correct
        correct = check_if_prediction_is_correct(y_pred, y_true)
        energy = mc_info[:, 2]
        particle_type = mc_info[:, 1]
        is_cc = mc_info[:, 3]

        ax = np.newaxis

        # make a temporary energy_correct array for this batch
        arr_energy_correct_temp = np.concatenate([energy[:, ax], correct[:, ax], particle_type[:, ax], is_cc[:, ax], y_pred], axis=1)

        if arr_energy_correct is None:
            arr_energy_correct = np.zeros((steps * batchsize, arr_energy_correct_temp.shape[1:2][0]), dtype=np.float32)
        arr_energy_correct[s*batchsize : (s+1) * batchsize] = arr_energy_correct_temp

    return arr_energy_correct


def check_if_prediction_is_correct(y_pred, y_true):
    """
    Checks if the predictions in y_pred are true.
    E.g. y_pred = [0.1, 0.1, 0.8] ; y_true = [0,0,1] -> Correct.
    Warning: There's a loophole if the prediction is not definite, e.g. y_pred = [0.4, 0.4, 0.2].
    :param ndarray(ndim=2) y_pred: 2D array that contains the predictions of a network on a number of events.
                                   Shape=(#events, n_classes).
    :param ndarray(ndim=2) y_true: 2D array that contains the true classes for the events. Shape=(#events, n_classes).
    :return: ndarray(ndim=1) correct: 1D array that specifies if the prediction for the single events is correct (True) or False.
    """
    # TODO loophole if pred has two or more max values per row
    class_pred = np.argmax(y_pred, axis=1)
    class_true = np.argmax(y_true, axis=1)

    correct = np.equal(class_pred, class_true)
    return correct

#------------- Functions used in evaluating the performance of model -------------#


#------------- Functions used in making Matplotlib plots -------------#

def make_energy_to_accuracy_plot(arr_energy_correct, title, filepath, plot_range=(3, 100)):
    """
    Makes a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    """
    # Calculate accuracy in energy range
    energy = arr_energy_correct[:, 0]
    correct = arr_energy_correct[:, 1]

    hist_1d_energy = np.histogram(energy, bins=98, range=plot_range)
    hist_1d_energy_correct = np.histogram(arr_energy_correct[correct == 1, 0], bins=98, range=plot_range)

    bin_edges = hist_1d_energy[1]
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_correct[0], hist_1d_energy[0], dtype=np.float32)
    # For making it work with matplotlib step plot
    hist_1d_energy_accuracy_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_bins))

    plt_bar_1d_energy_accuracy = plt.step(bin_edges, hist_1d_energy_accuracy_bins_leading_zero, where='pre')

    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.title(title)
    plt.grid(True)

    plt.savefig(filepath)


def make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct_classes, title, filename, plot_range=(3,100)):
    """
    Makes a mpl step plot of Energy vs. 'Fraction of events classified as track' for multiple classes.
    Till now only used for muon-CC vs elec-CC.
    :param ndarray arr_energy_correct_classes: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param str title: Title that should be used in the plot.
    :param str filename: Filename that should be used for saving the plot.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    """
    fig, axes = plt.subplots()

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'muon-CC', plot_range, linestyle='-', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'a_muon-CC', plot_range, linestyle='--', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'elec-CC', plot_range, linestyle='-', color='r', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'a_elec-CC', plot_range, linestyle='--', color='r', invert=True)

    axes.legend(loc='center right')

    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    #plt.ylabel('Accuracy')
    plt.ylabel('Fraction of events classified as track')
    plt.ylim((0, 1.05))
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0)

    plt.savefig(filename + '_3-100GeV.pdf')

    x_ticks_major = np.arange(0, 101, 5)
    plt.xticks(x_ticks_major)
    plt.xlim((0,40))
    plt.savefig(filename + '_3-40GeV.pdf')


def make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, particle_type, plot_range=(3,100), linestyle='-', color='b', invert=False):
    """
    Makes a mpl 1D step plot with Energy vs. Accuracy for a certain input class (e.g. a_muon-CC).
    :param ndarray arr_energy_correct_classes: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param mpl.axes axes: mpl axes object that refers to an existing plt.sublots object.
    :param dict particle_types_dict: Dictionary that contains a (particle_type, is_cc) [-> muon-CC!] tuple in order to classify the events.
    :param str particle_type: Particle type that should be plotted, e.g. 'a_muon-CC'.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param str linestyle: Specifies the mpl linestyle that should be used.
    :param str color: Specifies the mpl color that should be used for plotting the step.
    :param bool invert: If True, it inverts the y-axis which may be useful for plotting a 'Fraction of events classified as track' plot.
    """
    class_vector = particle_types_dict[particle_type]

    arr_energy_correct_class = select_class(arr_energy_correct_classes, class_vector=class_vector)
    energy_class = arr_energy_correct_class[:, 0]
    correct_class = arr_energy_correct_class[:, 1]

    hist_1d_energy_class = np.histogram(energy_class, bins=98, range=plot_range)
    hist_1d_energy_correct_class = np.histogram(arr_energy_correct_class[correct_class == 1, 0], bins=98, range=plot_range)

    bin_edges = hist_1d_energy_class[1]
    hist_1d_energy_accuracy_class_bins = np.divide(hist_1d_energy_correct_class[0], hist_1d_energy_class[0], dtype=np.float32) # TODO solve division by zero

    if invert is True: hist_1d_energy_accuracy_class_bins = np.absolute(hist_1d_energy_accuracy_class_bins - 1)

    # For making it work with matplotlib step plot
    hist_1d_energy_accuracy_class_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_class_bins))

    axes.step(bin_edges, hist_1d_energy_accuracy_class_bins_leading_zero, where='pre', linestyle=linestyle, color=color, label=particle_type, zorder=3)


def select_class(arr_energy_correct_classes, class_vector):
    """
    Selects the rows in an arr_energy_correct_classes array that correspond to a certain class_vector.
    :param arr_energy_correct_classes: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param (int, int) class_vector: Specifies the class that is used for filtering the array. E.g. (14,1) for muon-CC.
    """
    check_arr_for_class = arr_energy_correct_classes[:,2:4] == class_vector  # returns a bool for each of the class_vector entries

    # Select only the events, where every bool for one event is True
    indices_rows_with_class = np.logical_and(check_arr_for_class[:, 0], check_arr_for_class[:, 1])

    selected_rows_of_class = arr_energy_correct_classes[indices_rows_with_class]

    return selected_rows_of_class

# ------------- Functions used in making Matplotlib plots -------------#


def make_prob_hists(arr_energy_correct, modelname):
    """
    Function that makes (class-) probability histograms based on the arr_energy_correct.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, energy, ptype, is_cc, y_pred].
    :param str modelname: Name of the model that is used for saving the plots.
    """
    def configure_hstack_plot(plot_title, savepath):
        """
        Configure a mpl plot with GridLines, Logscale etc.
        :param str plot_title: Title that should be used for the plot.
        :param str savepath: path that should be used for saving the plot.
        """
        axes.legend(loc='upper center')
        plt.grid(True, zorder=0)
        #plt.yscale('log')

        x_ticks_major = np.arange(0, 1.1, 0.1)
        plt.xticks(x_ticks_major)
        plt.minorticks_on()

        plt.xlabel('Probability')
        plt.ylabel('Normed Quantity')
        title = plt.title(plot_title)
        title.set_position([.5, 1.04])

        plt.savefig(savepath)

    fig, axes = plt.subplots()
    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    # make energy cut, 3-40GeV
    arr_energy_correct_ecut = arr_energy_correct[arr_energy_correct[:, 0] <= 40]

    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'muon-CC', 0, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_muon-CC', 0, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'elec-CC', 0, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_elec-CC', 0, plot_range=(0, 1), color='r', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as elec-CC (shower)', savepath='results/plots/PT_hist1D_prob_shower_' + modelname + '.pdf')
    plt.cla()

    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'muon-CC', 1, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_muon-CC', 1, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'elec-CC', 1, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_elec-CC', 1, plot_range=(0, 1), color='r', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as muon-CC (track)', savepath='results/plots/PT_hist1D_prob_track_' + modelname + '.pdf')
    plt.cla()


def make_prob_hist_class(arr_energy_correct, axes, particle_types_dict, particle_type, prob_class_index, plot_range=(0,1), color='b', linestyle='-'):
    """
    Makes mpl hists based on an arr_energy_correct for a certain particle class (e.g. 'muon-CC').
    :param ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    :param mpl.axes axes: mpl axes object that refers to an existing plt.sublots object.
    :param dict particle_types_dict: Dictionary that contains a (particle_type, is_cc) [-> muon-CC!] tuple in order to classify the events.
    :param str particle_type: Particle type that should be plotted, e.g. 'a_muon-CC'.
    :param int prob_class_index: Specifies which class (e.g. elec-CC/muon-CC) should be used for the probability plots.
                                 E.g. for 2 classes: [1,0] -> shower -> index=0, [0,1] -> track -> index=1
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param str color: Specifies the mpl color that should be used for plotting the hist.
    :param str linestyle: Specifies the mpl linestyle that should be used.
    """
    ptype_vector = particle_types_dict[particle_type]
    arr_energy_correct_ptype = select_class(arr_energy_correct, class_vector=ptype_vector)

    # prob_class_index = 0/1, 0 is shower, 1 is track
    prob_ptype_class = arr_energy_correct_ptype[:, 4 + prob_class_index]

    hist_1d_prob_ptype_class = axes.hist(prob_ptype_class, bins=40, range=plot_range, normed=True, color=color, label=particle_type, histtype='step', linestyle=linestyle, zorder=3)





