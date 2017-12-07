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
        event_id = mc_info[:, 0]
        run_id = mc_info[:, 9]
        bjorken_y = mc_info[:, 4]

        ax = np.newaxis

        # make a temporary energy_correct array for this batch
        arr_energy_correct_temp = np.concatenate([energy[:, ax], correct[:, ax], particle_type[:, ax], is_cc[:, ax],
                                                  y_pred, event_id[:, ax], run_id[:, ax], bjorken_y[:, ax]], axis=1)

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

#-- Functions for applying Pheid precuts to the events --#

def add_pid_column_to_array(array, particle_type_dict, key):
    """
    Takes an array and adds two pid columns (particle_type, is_cc) to it along axis_1.
    :param ndarray(ndim=2) array: array to which the pid columns should be added.
    :param dict particle_type_dict: dict that contains the pid tuple (e.g. for muon-CC: (14,1)) for each interaction type at pos[1].
    :param str key: key of the dict that specifies which kind of pid tuple should be added to the array (dependent on interaction type).
    :return: ndarray(ndim=2) array_with_pid: array with additional pid columns. ordering: [pid_columns, array_columns]
    """
    # add pid columns particle_type, is_cc to events
    pid = np.array(particle_type_dict[key][1], dtype=np.float32).reshape((1,2))
    pid_array = np.repeat(pid, array.shape[0] , axis=0)

    array_with_pid = np.concatenate((pid_array, array), axis=1)
    return array_with_pid


def load_pheid_event_selection():
    """
    Loads the pheid event that survive the precuts from a .txt file, adds a pid column to them and returns it.
    :return: ndarray(ndim=2) arr_pheid_sel_events: 2D array that contains [particle_type, is_cc, event_id, run_id]
                                                   for each event that survives the precuts.
    """
    path = '/home/woody/capn/mppi033h/Code/HPC/cnns/results/plots/pheid_event_selection_txt/' # folder for storing the precut .txts

    # Moritz's precuts
    # particle_type_dict = {'muon-CC': ['muon_cc_3_100_selectedEvents_forMichael_fixed.txt', (14,1)],
    #                       'elec-CC': ['elec_cc_3_100_selectedEvents_forMichael_fixed.txt', (12,1)]}

    # Containment cut
    particle_type_dict = {'muon-CC': ['muon_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt', (14,1)],
                          'elec-CC': ['elec_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt', (12,1)]}

    arr_pheid_sel_events = None
    for key in particle_type_dict:
        txt_file = particle_type_dict[key][0]

        if arr_pheid_sel_events is None:
            arr_pheid_sel_events = np.loadtxt(path + txt_file, dtype=np.float32)
            arr_pheid_sel_events = add_pid_column_to_array(arr_pheid_sel_events, particle_type_dict, key)
        else:
            temp_pheid_sel_events = np.loadtxt(path + txt_file, dtype=np.float32)
            temp_pheid_sel_events = add_pid_column_to_array(temp_pheid_sel_events, particle_type_dict, key)

            arr_pheid_sel_events = np.concatenate((arr_pheid_sel_events, temp_pheid_sel_events), axis=0)

    # swap columns from run_id, event_id to event_id, run_id
    arr_pheid_sel_events[:, [2,3]] = arr_pheid_sel_events[:, [3,2]] # particle_type, is_cc, event_id, run_id

    return arr_pheid_sel_events


def in_nd(a, b, absolute=True, assume_unique=False):
    """
    Function that generalizes the np in_1d function to nd.
    Checks if entries in axis_0 of a exist in b and returns the bool array for all rows.
    Kind of hacky by using str views on the np arrays.
    :param ndarray(ndim=2) a: array where it should be checked whether each row exists in b or not.
    :param ndarray(ndim=2) b: array upon which the rows of a are checked.
    :param bool absolute: Specifies if absolute() should be called on the arrays before applying in_nd.
                     Useful when e.g. in_nd shouldn't care about particle (+) or antiparticle (-).
    :param bool assume_unique: ff True, the input arrays are both assumed to be unique, which can speed up the calculation.
    :return: ndarray(ndim=1): Boolean array that specifies for each row of a if it also exists in b or not.
    """
    if a.dtype!=b.dtype: raise TypeError('The dtype of array a must be equal to the dtype of array b.')
    a = np.asarray(a, order='C')
    b = np.asarray(b, order='C')

    if absolute is True: # we don't care about e.g. particles or antiparticles
        a = np.absolute(a)
        b = np.absolute(b)

    a = a.ravel().view((np.str, a.itemsize * a.shape[1]))
    b = b.ravel().view((np.str, b.itemsize * b.shape[1]))
    return np.in1d(a, b, assume_unique)


def arr_energy_correct_select_pheid_events(arr_energy_correct, invert=False):
    """
    Function that applies the Pheid precuts to an arr_energy_correct.
    :param ndarray(ndim=2) arr_energy_correct: array from the make_performance_array_energy_correct() function.
    :param bool invert: Instead of selecting all events that survive the Pheid precut, it _removes_ all the Pheid events
                        and leaves all the non-Pheid events.
    :return: ndarray(ndim=2) arr_energy_correct: same array, but after applying the Pheid precuts on it.
                                                 (events that don't survive the precuts are missing!)
    """
    pheid_evt_run_id = load_pheid_event_selection()

    evt_run_id_in_pheid = in_nd(arr_energy_correct[:, [2,3,6,7]], pheid_evt_run_id, absolute=True) # 2,3,6,7: particle_type, is_cc, event_id, run_id

    if invert is True: evt_run_id_in_pheid = np.invert(evt_run_id_in_pheid)

    arr_energy_correct = arr_energy_correct[evt_run_id_in_pheid] # apply boolean in_pheid selection to the array

    return arr_energy_correct

#-- Functions for applying Pheid precuts to the events --#

#-- Utility functions --#

def print_absolute_performance(arr_energy_correct, print_text='Performance: '):
    """
    Takes an arr_energy_correct, calculates the absolute performance of the predictions in the array and prints the result.
    :param ndarray(ndim=2) arr_energy_correct: array from the make_performance_array_energy_correct() function.
    :param str print_text: String that should be used in printing before printing the results.
    """
    correct = arr_energy_correct[:, 1] # select correct column
    n_correct = np.count_nonzero(correct, axis=0) # count how many times the predictions were True
    n_total = arr_energy_correct.shape[0] # count the total number of predictions
    performance = n_correct/float(n_total)
    print print_text
    print str(performance *100)
    print arr_energy_correct.shape

#-- Utility functions --#

#-- Functions for making energy to accuracy plots --#

def make_energy_to_accuracy_plot(arr_energy_correct, title, filepath, plot_range=(3, 100), compare_pheid=False):
    """
    Makes a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    :param bool compare_pheid: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots.
    """
    print_absolute_performance(arr_energy_correct, print_text='Performance and array_shape without Pheid event selection: ')
    if compare_pheid is True:
        arr_energy_correct = arr_energy_correct_select_pheid_events(arr_energy_correct, invert=False)
    print_absolute_performance(arr_energy_correct, print_text='Performance and array_shape with Pheid event selection: ')

    # Calculate accuracy in energy range
    energy = arr_energy_correct[:, 0]
    correct = arr_energy_correct[:, 1]

    hist_1d_energy = np.histogram(energy, bins=98, range=plot_range)
    hist_1d_energy_correct = np.histogram(arr_energy_correct[correct == 1, 0], bins=98, range=plot_range)

    bin_edges = hist_1d_energy[1]
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_correct[0], hist_1d_energy[0], dtype=np.float32)
    hist_1d_energy_accuracy_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_bins)) # For making it work with matplotlib step plot

    plt_bar_1d_energy_accuracy = plt.step(bin_edges, hist_1d_energy_accuracy_bins_leading_zero, where='pre', zorder=3)

    x_ticks_major = np.arange(0, 101, 10)
    y_ticks_major = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1.05))
    plt.yticks(y_ticks_major)
    plt.title(title)
    plt.grid(True, zorder=0, linestyle='dotted')

    plt.savefig(filepath + '.pdf')
    plt.savefig(filepath + '.png', dpi=600)


def make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct, title, filename, plot_range=(3, 100), compare_pheid=False):
    """
    Makes a mpl step plot of Energy vs. 'Fraction of events classified as track' for multiple classes.
    Till now only used for muon-CC vs elec-CC.
    :param ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param str title: Title that should be used in the plot.
    :param str filename: Filename that should be used for saving the plot.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param bool compare_pheid: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots.
    """
    print_absolute_performance(arr_energy_correct, print_text='Performance and array_shape without Pheid event selection: ')
    if compare_pheid is True:
        arr_energy_correct = arr_energy_correct_select_pheid_events(arr_energy_correct, invert=False)
        print_absolute_performance(arr_energy_correct, print_text='Performance and array_shape with Pheid event selection: ')

    fig, axes = plt.subplots()

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    make_step_plot_1d_energy_accuracy_class(arr_energy_correct, axes, particle_types_dict, 'muon-CC', plot_range, linestyle='-', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct, axes, particle_types_dict, 'a_muon-CC', plot_range, linestyle='--', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct, axes, particle_types_dict, 'elec-CC', plot_range, linestyle='-', color='r', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct, axes, particle_types_dict, 'a_elec-CC', plot_range, linestyle='--', color='r', invert=True)

    axes.legend(loc='center right')

    x_ticks_major = np.arange(0, 101, 10)
    y_ticks_major = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    #plt.ylabel('Accuracy')
    plt.ylabel('Fraction of events classified as track')
    plt.ylim((0, 1.05))
    plt.yticks(y_ticks_major)
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0, linestyle='dotted')

    plt.savefig(filename + '_3-100GeV.pdf')
    plt.savefig(filename + '_3-100GeV.png', dpi=600)

    x_ticks_major = np.arange(0, 101, 5)
    plt.xticks(x_ticks_major)
    plt.xlim((0,40))
    plt.savefig(filename + '_3-40GeV.pdf')
    plt.savefig(filename + '_3-40GeV.png', dpi=600)


def make_step_plot_1d_energy_accuracy_class(arr_energy_correct, axes, particle_types_dict, particle_type, plot_range=(3, 100), linestyle='-', color='b', invert=False):
    """
    Makes a mpl 1D step plot with Energy vs. Accuracy for a certain input class (e.g. a_muon-CC).
    :param ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param mpl.axes axes: mpl axes object that refers to an existing plt.sublots object.
    :param dict particle_types_dict: Dictionary that contains a (particle_type, is_cc) [-> muon-CC!] tuple in order to classify the events.
    :param str particle_type: Particle type that should be plotted, e.g. 'a_muon-CC'.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param str linestyle: Specifies the mpl linestyle that should be used.
    :param str color: Specifies the mpl color that should be used for plotting the step.
    :param bool invert: If True, it inverts the y-axis which may be useful for plotting a 'Fraction of events classified as track' plot.
    """
    class_vector = particle_types_dict[particle_type]

    arr_energy_correct_class = select_class(arr_energy_correct, class_vector=class_vector)
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


#-- Functions for making energy to accuracy plots --#

#-- Functions for making probability plots --#

def make_prob_hists(arr_energy_correct, modelname, compare_pheid=False):
    """
    Function that makes (class-) probability histograms based on the arr_energy_correct.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, energy, ptype, is_cc, y_pred].
    :param str modelname: Name of the model that is used for saving the plots.
    :param bool compare_pheid: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots.
    """
    def configure_hstack_plot(plot_title, savepath):
        """
        Configure a mpl plot with GridLines, Logscale etc.
        :param str plot_title: Title that should be used for the plot.
        :param str savepath: path that should be used for saving the plot.
        """
        axes.legend(loc='upper center')
        plt.grid(True, zorder=0, linestyle='dotted')
        #plt.yscale('log')

        x_ticks_major = np.arange(0, 1.1, 0.1)
        plt.xticks(x_ticks_major)
        plt.minorticks_on()

        plt.xlabel('Probability')
        plt.ylabel('Normed Quantity')
        title = plt.title(plot_title)
        title.set_position([.5, 1.04])

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png', dpi=600)

    if compare_pheid is True:
        arr_energy_correct = arr_energy_correct_select_pheid_events(arr_energy_correct, invert=False)

    fig, axes = plt.subplots()
    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    # make energy cut, 3-40GeV
    arr_energy_correct_ecut = arr_energy_correct[arr_energy_correct[:, 0] <= 40]

    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'muon-CC', 0, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_muon-CC', 0, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'elec-CC', 0, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_elec-CC', 0, plot_range=(0, 1), color='r', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as elec-CC (shower) 3-40GeV', savepath='results/plots/PT_hist1D_prob_shower_' + modelname)
    plt.cla()

    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'muon-CC', 1, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_muon-CC', 1, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'elec-CC', 1, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_elec-CC', 1, plot_range=(0, 1), color='r', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as muon-CC (track) 3-40GeV', savepath='results/plots/PT_hist1D_prob_track_' + modelname)
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


#-- Functions for making probability plots --#

#-- Functions for making property (e.g. bjorken_y) vs accuracy plots --#

def make_property_to_accuracy_plot(arr_energy_correct, property_type, title, filename, e_cut=False, compare_pheid=False):

    if compare_pheid is True:
        arr_energy_correct = arr_energy_correct_select_pheid_events(arr_energy_correct, invert=False)

    if e_cut is True: arr_energy_correct = arr_energy_correct[arr_energy_correct[:, 0] <= 40] # 3-40 GeV

    fig, axes = plt.subplots()

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}
    properties = {'bjorken-y': {'index': 8, 'n_bins': 10, 'plot_range': (0, 1)}}
    prop = properties[property_type]

    make_step_plot_1d_property_accuracy_class(prop, arr_energy_correct, axes, particle_types_dict, 'muon-CC', linestyle='-', color='b')
    make_step_plot_1d_property_accuracy_class(prop, arr_energy_correct, axes, particle_types_dict, 'a_muon-CC', linestyle='--', color='b')
    make_step_plot_1d_property_accuracy_class(prop, arr_energy_correct, axes, particle_types_dict, 'elec-CC', linestyle='-', color='r')
    make_step_plot_1d_property_accuracy_class(prop, arr_energy_correct, axes, particle_types_dict, 'a_elec-CC', linestyle='--', color='r')

    axes.legend(loc='center right')

    stepsize_x_ticks = (prop['plot_range'][1] - prop['plot_range'][0]) / float(prop['n_bins'])
    x_ticks_major = np.arange(prop['plot_range'][0], prop['plot_range'][1] + 0.1*prop['plot_range'][1], stepsize_x_ticks)
    y_ticks_major = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel(property_type)
    plt.ylabel('Accuracy')
    plt.ylim((0, 1.05))
    plt.yticks(y_ticks_major)
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0, linestyle='dotted')

    plt.savefig(filename + '_3-100GeV.pdf')
    plt.savefig(filename + '_3-100GeV.png', dpi=600)


def make_step_plot_1d_property_accuracy_class(prop, arr_energy_correct, axes, particle_types_dict, particle_type, linestyle='-', color='b'):

    class_vector = particle_types_dict[particle_type]

    arr_energy_correct_class = select_class(arr_energy_correct, class_vector=class_vector)
    property_class = arr_energy_correct_class[:, prop['index']]
    correct_class = arr_energy_correct_class[:, 1]

    hist_1d_property_class = np.histogram(property_class, bins=prop['n_bins'], range=prop['plot_range']) # all
    hist_1d_property_correct_class = np.histogram(arr_energy_correct_class[correct_class == 1, prop['index']], bins=prop['n_bins'], range=prop['plot_range']) # only the ones that were predicted correctly

    bin_edges = hist_1d_property_class[1]
    hist_1d_property_accuracy_class_bins = np.divide(hist_1d_property_correct_class[0], hist_1d_property_class[0], dtype=np.float32) # TODO solve division by zero

    # For making it work with matplotlib step plot
    hist_1d_property_accuracy_class_bins_leading_zero = np.hstack((0, hist_1d_property_accuracy_class_bins))

    axes.step(bin_edges, hist_1d_property_accuracy_class_bins_leading_zero, where='pre', linestyle=linestyle, color=color, label=particle_type, zorder=3)


#-- Functions for making property (e.g. bjorken_y) vs accuracy plots --#

#------------- Functions used in making Matplotlib plots -------------#








