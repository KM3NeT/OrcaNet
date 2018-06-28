#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code for the evaluation of a network's performance after training."""

import os
import h5py
import numpy as np
import keras as ks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .cnn_utilities import generate_batches_from_hdf5_file

#------------- Functions used in evaluating the performance of model -------------#

def get_nn_predictions_and_mc_info(model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, modelname, samples=None):
    """
    Creates an energy_correct array based on test_data that specifies for every event, if the model's prediction is True/False.
    :param ks.model.Model model: Fully trained Keras model of a neural network.
    :param str test_files: List that contains the test files. Format should be [ ( [], ), ... ].
    :param list(tuple) n_bins: The number of bins for each dimension (x,y,z,t) in the testfile. Can contain multiple n_bins tuples.
    :param (int, str) class_type: The number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for predicting.
    :param list(ndarray) xs_mean: mean_image(s) of the x dataset if zero-centering is enabled.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped in the generator.
    :param str str_ident: string identifier that is parsed to the generator. Needed for some projection types.
    :param None/int samples: Number of events that should be predicted. If samples=None, the whole file will be used.
    :param str modelname: name of the nn model.
    :return ndarray arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
    """
    # get total number of samples
    cum_number_of_steps = get_cum_number_of_steps(test_files, batchsize)
    ax = np.newaxis

    arr_nn_pred = None
    for f_number, f in enumerate(test_files):
        generator = generate_batches_from_hdf5_file(f[0], batchsize, n_bins, class_type, str_ident, zero_center_image=xs_mean, yield_mc_info=True, swap_col=swap_4d_channels) # f_size=samples prob not necessary

        if samples is None: samples = len(h5py.File(f[0][0], 'r')['y'])
        steps = samples/batchsize

        arr_nn_pred_row_start = cum_number_of_steps[f_number] * batchsize
        for s in xrange(steps):
            if s % 100 == 0: print 'Predicting in step ' + str(s) + ' on file ' + str(f_number)

            xs, y_true, mc_info = next(generator)
            y_pred = model.predict_on_batch(xs)

            if class_type[1] == 'energy_and_direction_and_bjorken-y': # y_pred and y_true is a list with the 1d arrays energy, dir, ...
                y_pred = np.concatenate(y_pred, axis=1)
                y_true = np.concatenate([label_arr[:, ax] for label_arr in y_true], axis=1)

            # check if the predictions were correct
            energy = mc_info[:, 2]
            particle_type = mc_info[:, 1]
            is_cc = mc_info[:, 3]
            event_id = mc_info[:, 0]
            run_id = mc_info[:, 9]
            bjorken_y = mc_info[:, 4]
            dir_x, dir_y, dir_z = mc_info[:, 5], mc_info[:, 6], mc_info[:, 7]

            # make a temporary energy_correct array for this batch
            arr_nn_pred_temp = np.concatenate([run_id[:, ax], event_id[:, ax], particle_type[:, ax], is_cc[:, ax], energy[:, ax],
                                               bjorken_y[:, ax], dir_x[:, ax], dir_y[:, ax], dir_z[:, ax], y_pred, y_true], axis=1)

            if arr_nn_pred is None: arr_nn_pred = np.zeros((cum_number_of_steps[-1] * batchsize, arr_nn_pred_temp.shape[1:2][0]), dtype=np.float32)
            arr_nn_pred[arr_nn_pred_row_start + s*batchsize : arr_nn_pred_row_start + (s+1) * batchsize] = arr_nn_pred_temp

    make_pred_h5_file(arr_nn_pred, filepath='predictions/' + modelname) # TODO check for empty lines

    return arr_nn_pred


def get_cum_number_of_steps(files, batchsize):
    """
    Function that calculates the cumulative number of prediction steps for the single files in the <files> list.
    Typically used during prediction when the data is split to multiple input files.
    :param list(tuple(list,int)) files: file list that should have the shape [ ( [], ), ... ]
    :param int batchsize: batchsize that is used during the prediction
    """
    cum_number_of_steps = [0]
    for f in files:
        samples = len(h5py.File(f[0][0], 'r')['y'])
        steps = samples/batchsize
        cum_number_of_steps.append(steps) # [0, steps_sample_1, steps_sample_1 + steps_sample_2, ...]

    return cum_number_of_steps


def make_pred_h5_file(arr_nn_pred, filepath, mc_prod='3-100GeV'):
    """
    Takes an arr_nn_pred for the track/shower classification and saves the important reco columns to a .h5 file.
    :param ndarray arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
    :param str mc_prod: optional parameter that specifies which mc prod is used. E.g. 3-100GeV or 1-5GeV.
    :param str filepath: filepath that should be used for saving the .h5 file.
    """
    #TODO fix for new
    arr_nn_output = np.concatenate([arr_nn_pred[:, 0:1], arr_nn_pred[:, 1:2], arr_nn_pred[:, 2:3], arr_nn_pred[:, 3:4], arr_nn_pred[:, 10:11]], axis=1)

    f = h5py.File(filepath + mc_prod + '.h5', 'w')
    dset = f.create_dataset('nn_output', data=arr_nn_output)
    dset.attrs['array_contents'] = 'Columns: run_id, event_id, PID (particle_type + is_cc), y_pred_track. \n' \
                                   'PID info: (12, 0): elec-NC, (12, 1): elec-CC, (14, 1): muon-CC, (16, 1): tau-CC \n' \
                                   'y_pred_track info: probability of the neural network for this event to be a track'
    f.close()


#------------- Functions used in evaluating the performance of model -------------#


#------------- Functions used in making Matplotlib plots -------------#

#-- Functions for applying Pheid precuts to the events --#

def add_pid_column_to_array(array, particle_type_dict, key):
    """
    Takes an array and adds two pid columns (particle_type, is_cc) to it along axis_1.
    :param ndarray(ndim=2) array: array to which the pid columns should be added.
    :param dict particle_type_dict: dict that contains the pid tuple (e.g. for muon-CC: (14,1)) for each interaction type at pos[1].
    :param str key: key of the dict that specifies which kind of pid tuple should be added to the array (dependent on interaction type).
    :return ndarray(ndim=2) array_with_pid: array with additional pid columns. ordering: [array_columns, pid_columns]
    """
    # add pid columns particle_type, is_cc to events
    pid = np.array(particle_type_dict[key][1], dtype=np.float32).reshape((1,2))
    pid_array = np.repeat(pid, array.shape[0] , axis=0)

    array_with_pid = np.concatenate((array, pid_array), axis=1)
    return array_with_pid


def load_pheid_event_selection(precuts='3-100_GeV_prod'):
    """
    Loads the pheid event that survive the precuts from a .txt file, adds a pid column to them and returns it.
    :param str precuts: specifies, which precuts should be loaded.
    :return ndarray(ndim=2) arr_pheid_sel_events: 2D array that contains [particle_type, is_cc, event_id, run_id]
                                                   for each event that survives the precuts.
    """
    path = 'results/plots/pheid_event_selection_txt/' # folder for storing the precut .txts

    #### Precuts
    if precuts == '3-100_GeV_prod':
        # 3-100 GeV
        particle_type_dict = {'muon-CC': ['muon_cc_3_100_selectedEvents_forMichael_01_18.txt', (14,1)],
                              'elec-CC': ['elec_cc_3_100_selectedEvents_forMichael_01_18.txt', (12,1)],
                              'elec-NC': ['elec_nc_3_100_selectedEvents_forMichael.txt', (12, 0)],
                              'tau-CC': ['tau_cc_3_100_selectedEvents_forMichael.txt', (16, 1)]}

    elif precuts == '1-5_GeV_prod':
        # 1-5 GeV
        particle_type_dict = {'muon-CC': ['muon_cc_1_5_selectedEvents_forMichael.txt', (14,1)],
                              'elec-CC': ['elec_cc_1_5_selectedEvents_forMichael.txt', (12,1)],
                              'elec-NC': ['elec_nc_1_5_selectedEvents_forMichael.txt', (12, 0)]}

    elif precuts == '3-100_GeV_containment_cut':
        # 3-100 GeV Containment cut
        particle_type_dict = {'muon-CC': ['muon_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt', (14,1)],
                              'elec-CC': ['elec_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt', (12,1)]}

    elif precuts == '3-100_GeV_prod_energy_comparison':
        path = '/home/woody/capn/mppi033h/Data/various/'
        particle_type_dict = {'muon-CC': ['cuts_shallow_3_100_muon_cc.txt', (14,1)],
                              'elec-CC': ['cuts_shallow_3_100_elec_cc.txt', (12,1)]}

    elif precuts == '3-100_GeV_prod_energy_comparison_is_good':
        path = '/home/woody/capn/mppi033h/Data/various/'
        particle_type_dict = {'muon-CC': ['cuts_shallow_3_100_muon_cc_is_good.txt', (14,1)],
                              'elec-CC': ['cuts_shallow_3_100_elec_cc_is_good.txt', (12,1)]}

    else:
        raise ValueError('The specified precuts option "' + str(precuts) + '" is not available.')

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
    a, b = np.asarray(a, order='C'), np.asarray(b, order='C')

    if absolute is True: # we don't care about e.g. particles or antiparticles
        a, b = np.absolute(a), np.absolute(b)

    a = a.ravel().view((np.str, a.itemsize * a.shape[1]))
    b = b.ravel().view((np.str, b.itemsize * b.shape[1]))
    return np.in1d(a, b, assume_unique)


def arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts='3-100_GeV_prod'):
    """
    Function that applies the Pheid precuts to an arr_nn_pred.
    :param ndarray(ndim=2) arr_nn_pred: array from the get_nn_predictions_and_mc_info() function.
    :param bool invert: Instead of selecting all events that survive the Pheid precut, it _removes_ all the Pheid events
                        and leaves all the non-Pheid events.
    :param str precuts: specifies, which precuts should be loaded.
    :return ndarray(ndim=2) arr_nn_pred: same array, but after applying the Pheid precuts on it.
                                                 (events that don't survive the precuts are missing!)
    """
    pheid_evt_run_id = load_pheid_event_selection(precuts=precuts)

    evt_run_id_in_pheid = in_nd(arr_nn_pred[:, [0, 1, 2, 3]], pheid_evt_run_id, absolute=True)  # 0,1,2,3: run_id, event_id, particle_type, is_cc

    if invert is True: evt_run_id_in_pheid = np.invert(evt_run_id_in_pheid)

    arr_nn_pred = arr_nn_pred[evt_run_id_in_pheid] # apply boolean in_pheid selection to the array

    return arr_nn_pred

#-- Functions for applying Pheid precuts to the events --#

#-- Utility functions --#

def print_absolute_performance(arr_energy_correct, print_text='Performance: '):
    """
    For PID.
    Takes an array with energy and correct information, calculates the absolute performance of the predictions in the array and prints the result.
    :param ndarray(ndim=2) arr_energy_correct: array from the get_nn_predictions_and_mc_info() function.
    :param str print_text: String that should be used in printing before printing the results.
    """
    correct = arr_energy_correct[:, 13] # select correct column
    n_correct = np.count_nonzero(correct, axis=0) # count how many times the predictions were True
    n_total = arr_energy_correct.shape[0] # count the total number of predictions
    performance = n_correct/float(n_total)
    print print_text, '\n', str(performance *100), '\n', arr_energy_correct.shape

#-- Utility functions --#

#-- Functions for making energy to accuracy plots --#

#- Classification -#

def make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title, filename, plot_range=(3, 100), precuts=(False, '3-100_GeV_prod'), corr_cut_pred_0=0.5):
    """
    Makes a mpl step plot of Energy vs. 'Fraction of events classified as track' for multiple classes.
    Till now only used for muon-CC vs elec-CC.
    :param ndarray arr_nn_pred: array that contains at least the energy, correct, particle_type, is_cc and y_pred info for each event.
    :param str title: Title that should be used in the plot.
    :param str filename: Filename that should be used for saving the plot.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param tuple precuts: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots. # TODO fix docs
    :param float corr_cut_pred_0: sets the threshold for when an event is classified as class_0.
    """
    arr_nn_pred = get_correct_info_pid(arr_nn_pred, corr_cut_pred_0=corr_cut_pred_0)

    print_absolute_performance(arr_nn_pred, print_text='Performance and array_shape without Pheid event selection: ')
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, precuts=precuts[1], invert=False)
        print_absolute_performance(arr_nn_pred, print_text='Performance and array_shape with Pheid event selection: ')

    fig, axes = plt.subplots()

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1),
                           'a_elec-CC': (-12, 1), 'elec-NC': (12, 0), 'a_elec-NC': (-12, 0),
                           'tau-CC': (16, 1), 'a_tau-CC': (-16, 1)}

    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'muon-CC', plot_range, linestyle='-', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'a_muon-CC', plot_range, linestyle='--', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'elec-CC', plot_range, linestyle='-', color='r', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'a_elec-CC', plot_range, linestyle='--', color='r', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'elec-NC', plot_range, linestyle='-', color='saddlebrown', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'a_elec-NC', plot_range, linestyle='--', color='saddlebrown', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'tau-CC', plot_range, linestyle='-', color='g', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, 'a_tau-CC', plot_range, linestyle='--', color='g', invert=True)

    axes.legend(loc='center right', ncol=2)

    x_ticks_major = np.arange(0, 101, 10)
    y_ticks_major = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    plt.ylabel('Fraction of events classified as track')
    plt.ylim((0, 1.05))
    plt.yticks(y_ticks_major)
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0, linestyle='dotted')

    plt.text(0.05, 0.92, 'KM3NeT Preliminary', transform=axes.transAxes, weight='bold')

    plt.savefig(filename + '_3-100GeV.pdf')
    plt.savefig(filename + '_3-100GeV.png', dpi=600)

    x_ticks_major = np.arange(0, 101, 5)
    plt.xticks(x_ticks_major)
    plt.xlim((0,40))
    plt.savefig(filename + '_3-40GeV.pdf')
    plt.savefig(filename + '_3-40GeV.png', dpi=600)


def get_correct_info_pid(arr_nn_pred, corr_cut_pred_0=0.5):
    """
    Function that decides if the predictions by the neural network are correct or not for the binary pid classification.
    :param ndarray arr_nn_pred: Array that contains at least the energy, correct, particle_type, is_cc and y_pred info for each event.
    :param float corr_cut_pred_0: sets the threshold for when an event is classified as class_0.
    :return: ndarray arr_nn_pred: now has an additional correct column.
    """

    y_pred = arr_nn_pred[:, 9:11]
    y_true = arr_nn_pred[:, 11:13]
    correct = check_if_prediction_is_correct(y_pred, y_true, threshold_pred_0=corr_cut_pred_0)
    arr_nn_pred = np.concatenate([arr_nn_pred, correct[:, np.newaxis]], axis=1)

    return arr_nn_pred


def check_if_prediction_is_correct(y_pred, y_true, threshold_pred_0=0.5):
    """
    For track-shower: neuron_0 = shower, neuron_1 = track.
    If y_pred[:, 0] is larger than the threshold (threshold_pred_0), the event is classified as label 0 (-> returns True for being label 0).
    """
    if y_pred.shape[1] > 2: raise ValueError('The check if a prediction of the nn is correct is only available for binary categorization problems'
                                             'and not for problems with more than two classes!')
    binary_is_pred_0_pred = (y_pred[:, 0] > threshold_pred_0).astype('int') # binary true or false for every event (axis_0) in the y_pred array
    binary_is_pred_0_true = (y_true[:, 0] > 0).astype('int') # check in the y_true labels if the first pred class is True or not.

    correct = binary_is_pred_0_pred == binary_is_pred_0_true
    return correct

#TODO add KM3Net preliminary
def make_step_plot_1d_energy_accuracy_class(arr_nn_pred, axes, particle_types_dict, particle_type, plot_range=(3, 100), linestyle='-', color='b', invert=False):
    """
    Makes a mpl 1D step plot with Energy vs. Accuracy for a certain input class (e.g. a_muon-CC).
    :param ndarray arr_nn_pred: rray that contains at least the energy, correct, particle_type, is_cc and y_pred info for each event.
    :param mpl.axes axes: mpl axes object that refers to an existing plt.sublots object.
    :param dict particle_types_dict: Dictionary that contains a (particle_type, is_cc) [-> muon-CC!] tuple in order to classify the events.
    :param str particle_type: Particle type that should be plotted, e.g. 'a_muon-CC'.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param str linestyle: Specifies the mpl linestyle that should be used.
    :param str color: Specifies the mpl color that should be used for plotting the step.
    :param bool invert: If True, it inverts the y-axis which may be useful for plotting a 'Fraction of events classified as track' plot.
    """
    class_vector = particle_types_dict[particle_type]

    arr_nn_pred_class = select_class(arr_nn_pred, class_vector=class_vector)

    if arr_nn_pred_class.size == 0: return

    energy_class = arr_nn_pred_class[:, 4]
    correct_class = arr_nn_pred_class[:, 13]

    hist_1d_energy_class = np.histogram(energy_class, bins=97, range=plot_range)
    hist_1d_energy_correct_class = np.histogram(arr_nn_pred_class[correct_class == 1, 4], bins=97, range=plot_range)

    bin_edges = hist_1d_energy_class[1]
    hist_1d_energy_accuracy_class_bins = np.divide(hist_1d_energy_correct_class[0], hist_1d_energy_class[0], dtype=np.float32) # TODO solve division by zero

    if invert is True: hist_1d_energy_accuracy_class_bins = np.absolute(hist_1d_energy_accuracy_class_bins - 1)

    # For making it work with matplotlib step plot
    hist_1d_energy_accuracy_class_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_class_bins))

    label = {'muon-CC': r'$\nu_{\mu}-CC$', 'a_muon-CC': r'$\overline{\nu}_{\mu}-CC$', 'elec-CC': r'$\nu_{e}-CC$', 'a_elec-CC': r'$\overline{\nu}_{e}-CC$',
             'elec-NC': r'$\nu_{e}-NC$', 'a_elec-NC': r'$\overline{\nu}_{e}-NC$', 'tau-CC': r'$\nu_{\tau}-CC$', 'a_tau-CC': r'$\overline{\nu}_{\tau}-CC$'}
    axes.step(bin_edges, hist_1d_energy_accuracy_class_bins_leading_zero, where='pre', linestyle=linestyle, color=color, label=label[particle_type], zorder=3)


def select_class(arr_nn_pred_classes, class_vector):
    """
    Selects the rows in an arr_nn_pred_classes array that correspond to a certain class_vector.
    :param arr_nn_pred_classes: array that contains important information for each event (mc_info + model predictions).
    :param (int, int) class_vector: Specifies the class that is used for filtering the array. E.g. (14,1) for muon-CC.
    """
    check_arr_for_class = arr_nn_pred_classes[:, 2:4] == class_vector  # returns a bool for each of the class_vector entries

    # Select only the events, where every bool for one event is True
    indices_rows_with_class = np.logical_and(check_arr_for_class[:, 0], check_arr_for_class[:, 1])
    selected_rows_of_class = arr_nn_pred_classes[indices_rows_with_class]

    return selected_rows_of_class


#-- Functions for making energy to accuracy plots --#

#-- Functions for making probability plots --#

def make_prob_hists(arr_nn_pred, modelname, precuts=(False, '3-100_GeV_prod')):
    """
    Function that makes (class-) probability histograms based on the arr_nn_pred.
    :param ndarray(ndim=2) arr_nn_pred: 2D array that contains important information for each event (mc_info + model predictions).
    :param str modelname: Name of the model that is used for saving the plots.
    :param tuple precuts: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots. # TODO fix docs
    """
    def configure_hstack_plot(plot_title, savepath):
        """
        Configure a mpl plot with GridLines, Logscale etc.
        :param str plot_title: Title that should be used for the plot.
        :param str savepath: path that should be used for saving the plot.
        """
        axes.legend(loc='upper center', ncol=2)
        plt.grid(True, zorder=0, linestyle='dotted')
        #plt.yscale('log')

        x_ticks_major = np.arange(0, 1.1, 0.1)
        plt.xticks(x_ticks_major)
        plt.minorticks_on()

        plt.xlabel('OrcaNet probability')
        plt.ylabel('Normed Quantity')
        title = plt.title(plot_title)
        title.set_position([.5, 1.04])

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png', dpi=600)

    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, precuts=precuts[1], invert=False)

    fig, axes = plt.subplots()
    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1),
                           'a_elec-CC': (-12, 1), 'elec-NC': (12, 0), 'a_elec-NC': (-12, 0),
                           'tau-CC': (16, 1), 'a_tau-CC': (-16, 1)}

    # make energy cut, 3-40GeV
    arr_nn_pred_ecut = arr_nn_pred[arr_nn_pred[:, 4] <= 40]

    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'muon-CC', 0, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_muon-CC', 0, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'elec-CC', 0, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_elec-CC', 0, plot_range=(0, 1), color='r', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'elec-NC', 0, plot_range=(0,1), color='saddlebrown', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_elec-NC', 0, plot_range=(0, 1), color='saddlebrown', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'tau-CC', 0, plot_range=(0,1), color='g', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_tau-CC', 0, plot_range=(0, 1), color='g', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as shower, 3-40GeV', savepath='results/plots/1d/track_shower/ts_prob_shower_' + modelname)
    plt.cla()

    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'muon-CC', 1, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_muon-CC', 1, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'elec-CC', 1, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_elec-CC', 1, plot_range=(0, 1), color='r', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'elec-NC', 1, plot_range=(0,1), color='saddlebrown', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_elec-NC', 1, plot_range=(0, 1), color='saddlebrown', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'tau-CC', 1, plot_range=(0,1), color='g', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_tau-CC', 1, plot_range=(0, 1), color='g', linestyle='--')


    configure_hstack_plot(plot_title='Probability to be classified as track, 3-40GeV', savepath='results/plots/1d/track_shower/ts_prob_track_' + modelname)
    plt.cla()


def make_prob_hist_class(arr_nn_pred, axes, particle_types_dict, particle_type, prob_class_index, plot_range=(0, 1), color='b', linestyle='-'):
    """
    Makes mpl hists based on an arr_nn_pred for a certain particle class (e.g. 'muon-CC').
    :param ndarray arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
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
    arr_nn_pred_ptype = select_class(arr_nn_pred, class_vector=ptype_vector)

    if arr_nn_pred_ptype.size == 0: return

    # prob_class_index = 0/1, 0 is shower, 1 is track
    prob_ptype_class = arr_nn_pred_ptype[:, 9 + prob_class_index]

    label = {'muon-CC': r'$\nu_{\mu}-CC$', 'a_muon-CC': r'$\overline{\nu}_{\mu}-CC$', 'elec-CC': r'$\nu_{e}-CC$', 'a_elec-CC': r'$\overline{\nu}_{e}-CC$',
             'elec-NC': r'$\nu_{e}-NC$', 'a_elec-NC': r'$\overline{\nu}_{e}-NC$', 'tau-CC': r'$\nu_{\tau}-CC$', 'a_tau-CC': r'$\overline{\nu}_{\tau}-CC$'}

    axes.hist(prob_ptype_class, bins=40, range=plot_range, density=True, color=color, label=label[particle_type], histtype='step', linestyle=linestyle, zorder=3)


#-- Functions for making probability plots --#

#-- Functions for making property (e.g. bjorken_y) vs accuracy plots --#

# TODO doesn't work atm due to changed arr_nn_pred indices
def make_property_to_accuracy_plot(arr_nn_pred, property_type, title, filename, e_cut=False, precuts=(False, '3-100_GeV_prod')):
    """
    Function that makes property (e.g. energy) vs accuracy plots.
    :param ndarray(ndim=2) arr_nn_pred: Array that contains the energy, correct, particle_type, is_cc,... and y_pred info for each event.
    :param str property_type: Specifies which property should be plotted. Currently available: 'bjorken-y'.
    :param str title: Title of the plots.
    :param str filename: Full filepath for saving the plots.
    :param bool e_cut: Specifies if an energy cut should be used. If True, only events from 3-40GeV are selected for the plots.
    :param tuple precuts: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots. # TODO fix docs
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, precuts=precuts[1], invert=False)

    if e_cut is True: arr_nn_pred = arr_nn_pred[arr_nn_pred[:, 0] <= 40] # 3-40 GeV

    fig, axes = plt.subplots()

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1),
                           'a_elec-CC': (-12, 1), 'elec-NC': (12, 0), 'a_elec-NC': (-12, 0),
                           'tau-CC': (16, 1), 'a_tau-CC': (-16, 1)}
    properties = {'bjorken-y': {'index': 8, 'n_bins': 10, 'plot_range': (0, 1)}}
    prop = properties[property_type]

    make_step_plot_1d_property_accuracy_class(prop, arr_nn_pred, axes, particle_types_dict, 'muon-CC', linestyle='-', color='b')
    make_step_plot_1d_property_accuracy_class(prop, arr_nn_pred, axes, particle_types_dict, 'a_muon-CC', linestyle='--', color='b')
    make_step_plot_1d_property_accuracy_class(prop, arr_nn_pred, axes, particle_types_dict, 'elec-CC', linestyle='-', color='r')
    make_step_plot_1d_property_accuracy_class(prop, arr_nn_pred, axes, particle_types_dict, 'a_elec-CC', linestyle='--', color='r')

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

# TODO doesn't work atm due to changed arr_nn_pred indices
def make_step_plot_1d_property_accuracy_class(prop, arr_energy_correct, axes, particle_types_dict, particle_type, linestyle='-', color='b'):
    """
    Function for making 1D step plots property vs Accuracy.
    :param prop:
    :param ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    :param axes: mpl axes of the subplots. The step plot will be plotted onto the axes object.
    :param dict particle_types_dict: Dictionary that contains the tuple identifiers for different particle types, e.g. 'muon-CC': (14, 1).
    :param str particle_type: Particle type that should be selected, e.g. 'muon-CC'
    :param str linestyle: mpl linestyle used for the plot.
    :param str color: mpl color used for the plot.
    """
    class_vector = particle_types_dict[particle_type]

    arr_energy_correct_class = select_class(arr_energy_correct, class_vector=class_vector)

    if arr_energy_correct_class.size == 0: return

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

#TODO fix arr_nn_pred indices, doesn't work as of now
def make_hist_2d_property_vs_property(arr_nn_pred, modelname, property_types=('bjorken-y', 'probability'), e_cut=(3, 100), precuts=(False, '3-100_GeV_prod')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param property_types:
    :param e_cut:
    :param precuts:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, precuts=precuts[1], invert=False)

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1),
                           'a_elec-CC': (-12, 1), 'elec-NC': (12, 0), 'a_elec-NC': (-12, 0),
                           'tau-CC': (16, 1), 'a_tau-CC': (-16, 1)}
    properties = {'bjorken-y': {'index': 5, 'n_bins': 20, 'label': 'Bjorken-y'},
                  'probability': {'index': 10, 'n_bins': 20, 'label': 'Probability for being a track'}} # probability: index 4 -> elec-CC, index 5 -> muon-CC
    prop_1, prop_2 = properties[property_types[0]], properties[property_types[1]]


    for key in iter(particle_types_dict.keys()):
        make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut, key,
                           savepath='results/plots/2d/track_shower/hist_2d_' + key + '_' + property_types[0] + '_vs_'
                                    + property_types[1] + '_e_cut_' + str(e_cut[0]) + '_' + str(e_cut[1]) + '_' + modelname)

    # make multiple cuts in range
    e_cut_range, stepsize = (3, 40), 1

    pdf_pages = {}

    for key in iter(particle_types_dict.keys()):
        pdf_pages[key] = mpl.backends.backend_pdf.PdfPages('results/plots/2d/track_shower/hist_2d_' + key + '_'
                                                         + property_types[0] + '_vs_' + property_types[1] + '_e_cut_'
                                                           + str(e_cut[0]) + '_' + str(e_cut[1]) + '_multiple_' + modelname + '.pdf')

    for i in xrange(e_cut_range[1] - e_cut_range[0]):
        e_cut_temp = (e_cut_range[0] + i*stepsize, e_cut_range[0] + i * stepsize + stepsize)
        for key in iter(particle_types_dict.keys()):
            make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut_temp, key, pdf_file=pdf_pages[key])

    for i in xrange(e_cut_range[1] - e_cut_range[0]):
        e_cut_temp = (e_cut_range[0] + i*stepsize, e_cut_range[0] + i * stepsize + stepsize)
        for key in iter(particle_types_dict.keys()):
            make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut_temp, key, pdf_file=pdf_pages[key], log=True)

    for key in iter(pdf_pages.keys()):
        pdf_pages[key].close()

#TODO fix arr_nn_pred indices, doesn't work as of now
def make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut, particle_type, savepath='', pdf_file=None, log=False):
    """

    :param prop_1:
    :param prop_2:
    :param arr_nn_pred:
    :param particle_types_dict:
    :param e_cut:
    :param particle_type:
    :param savepath:
    :param pdf_file:
    :param log:
    :return:
    """
    arr_nn_pred = arr_nn_pred[np.logical_and(e_cut[0] <= arr_nn_pred[:, 4], arr_nn_pred[:, 4] <= e_cut[1])]

    class_vector = particle_types_dict[particle_type]

    arr_nn_pred_class = select_class(arr_nn_pred, class_vector=class_vector)

    if arr_nn_pred_class.size == 0: return

    property_1_class = arr_nn_pred_class[:, prop_1['index']]
    property_2_class = arr_nn_pred_class[:, prop_2['index']]

    fig = plt.figure()

    if log is True:
        plt.hist2d(property_1_class, property_2_class, bins=[prop_1['n_bins'], prop_2['n_bins']], norm=mpl.colors.LogNorm())
    else:
        plt.hist2d(property_1_class, property_2_class, bins=[prop_1['n_bins'], prop_2['n_bins']])

    plt.colorbar()
    plt.xlabel(prop_1['label'])
    plt.ylabel(prop_2['label'])

    plt.title(particle_type + ', ' + str(e_cut[0]) + '-' + str(e_cut[1]) + ' GeV \n' +
              prop_1['label'] + ' vs ' + prop_2['label'])

    if pdf_file is None:
        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png', dpi=600)

    else:
        pdf_file.savefig(fig)

    plt.clf()
    plt.close()

# TODO add KM3NeT preliminary
def calculate_and_plot_separation_pid(arr_nn_pred, modelname, precuts=(False, '3-100_GeV_prod')):
    """
    Calculates and plots the separability (1-c) plot.
    :param ndarray(ndim=2) arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
    :param str modelname: Name of the model used for plot savenames.
    :param tuple precuts: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots. # TODO fix docs
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, precuts=precuts[1], invert=False)

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    bins=40
    correlation_coefficients = []
    e_cut_range = np.logspace(0.3, 2, 18)

    n = 0
    for e_cut_temp in zip(e_cut_range[:-1], e_cut_range[1:]):
        n += 1
        if n <= 2: continue # ecut steffen

        arr_nn_pred_e_cut = arr_nn_pred[np.logical_and(e_cut_temp[0] <= arr_nn_pred[:, 4], arr_nn_pred[:, 4] <= e_cut_temp[1])]

        arr_nn_pred_e_cut_muon_cc = select_class(arr_nn_pred_e_cut, class_vector=particle_types_dict['muon-CC'])
        arr_nn_pred_e_cut_a_muon_cc = select_class(arr_nn_pred_e_cut, class_vector=particle_types_dict['a_muon-CC'])
        arr_nn_pred_e_cut_sum_muon_cc = np.concatenate([arr_nn_pred_e_cut_muon_cc, arr_nn_pred_e_cut_a_muon_cc], axis=0)

        arr_nn_pred_e_cut_elec_cc = select_class(arr_nn_pred_e_cut, class_vector=particle_types_dict['elec-CC'])
        arr_nn_pred_e_cut_a_elec_cc = select_class(arr_nn_pred_e_cut, class_vector=particle_types_dict['a_elec-CC'])
        arr_nn_pred_e_cut_sum_elec_cc = np.concatenate([arr_nn_pred_e_cut_elec_cc, arr_nn_pred_e_cut_a_elec_cc], axis=0)

        hist_prob_track_e_cut_sum_muon_cc = np.histogram(arr_nn_pred_e_cut_sum_muon_cc[:, 10], bins=40, density=True)
        hist_prob_track_e_cut_sum_elec_cc = np.histogram(arr_nn_pred_e_cut_sum_elec_cc[:, 10], bins=40, density=True)

        correlation_coeff_enumerator = 0
        for j in xrange(bins - 1):
            correlation_coeff_enumerator += hist_prob_track_e_cut_sum_muon_cc[0][j] * hist_prob_track_e_cut_sum_elec_cc[0][j]

        sum_prob_muon_cc = np.sum(hist_prob_track_e_cut_sum_muon_cc[0] ** 2)
        sum_prob_elec_cc = np.sum(hist_prob_track_e_cut_sum_elec_cc[0] ** 2)

        correlation_coeff_denominator = np.sqrt(sum_prob_muon_cc * sum_prob_elec_cc)

        correlation_coeff = 1 - correlation_coeff_enumerator/float(correlation_coeff_denominator)

        average_energy = 10**((np.log10(e_cut_temp[1]) + np.log10(e_cut_temp[0])) / float(2))

        correlation_coefficients.append((correlation_coeff, average_energy))

    correlation_coefficients = np.array(correlation_coefficients)
    # plot the array

    fig, axes = plt.subplots()
    plt.plot(correlation_coefficients[:, 1], correlation_coefficients[:, 0], 'b', marker='o', lw=0.5, markersize=3, label='Deep Learning')

    # data Steffen
    correlation_coefficients_steffen = np.array([(0.11781, 2.23872), (0.135522, 2.81838), (0.15929, 3.54813), (0.189369, 4.46684), (0.241118, 5.62341), (0.337265, 7.07946),
                                        (0.511258, 8.91251), (0.715669, 11.2202), (0.84403, 14.1254), (0.893825, 17.7828), (0.920772, 22.3872), (0.930504, 28.1838),
                                        (0.941481, 35.4813), (0.950237, 44.6684), (0.954144, 56.2341), (0.960136, 70.7946), (0.96377, 89.1251)])

    plt.plot(correlation_coefficients_steffen[:, 1], correlation_coefficients_steffen[:, 0], 'r', marker='o', lw=0.5, markersize=3, label='Shallow Learning')

    plt.xlabel('Energy [GeV]')
    plt.ylabel('Separability (1-c)')
    plt.grid(True, zorder=0, linestyle='dotted')

    axes.legend(loc='center right')
    title = plt.title('Separability for track vs shower PID')
    title.set_position([.5, 1.04])

    plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xticks(np.arange(0, 110, 10))
    plt.xscale('log')
    plt.text(0.05, 0.92, 'KM3NeT Preliminary', transform=axes.transAxes, weight='bold')

    plt.savefig('results/plots/1d/track_shower/Correlation_Coefficients_' + modelname + '.pdf')
    plt.savefig('results/plots/1d/track_shower/Correlation_Coefficients_' + modelname + '.png', dpi=600)

    plt.close()


#- Classification -#

#- Regression -#

def make_2d_energy_resolution_plot(arr_nn_pred, modelname, energy_bins=np.arange(3,101,1), compare_pheid=(False, '3-100_GeV_prod'), correct_energy=(False, 'median')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param energy_bins:
    :param compare_pheid:
    :param correct_energy:
    :return:
    """
    if compare_pheid[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=compare_pheid[1])
    if correct_energy[0] is True:
        arr_nn_pred = correct_reco_energy(arr_nn_pred, metric=correct_energy[1])

    energy_mc = arr_nn_pred[:, 4]
    energy_pred = arr_nn_pred[:, 9]
    is_track, is_shower = get_boolean_track_and_shower_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3])

    hist_2d_energy_track = np.histogram2d(energy_mc[is_track], energy_pred[is_track], energy_bins)
    hist_2d_energy_shower = np.histogram2d(energy_mc[is_shower], energy_pred[is_shower], energy_bins)
    bin_edges_energy = hist_2d_energy_shower[1] # doesn't matter if we take shower/track and x/y, bin edges are same for all of them

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/2d/energy/energy_resolution_' + modelname + '.pdf')

    # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
    # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
    energy_res_track = ax.pcolormesh(bin_edges_energy, bin_edges_energy, hist_2d_energy_track[0].T,
                                     norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_energy_track[0].T.max()))

    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
    title = plt.title(reco_name + 'Track like events (' + r'$\nu_{\mu}-CC$)')
    title.set_position([.5, 1.04])
    cbar1 = fig.colorbar(energy_res_track, ax=ax)
    cbar1.ax.set_ylabel('Number of events')
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Reconstructed energy (GeV)')
    plt.tight_layout()

    pdf_plots.savefig(fig)
    cbar1.remove()
    energy_res_track.remove()

    energy_res_shower = ax.pcolormesh(bin_edges_energy, bin_edges_energy, hist_2d_energy_shower[0].T, norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_energy_shower[0].T.max()))
    plt.title(reco_name + 'Shower like events (' + r'$\nu_{e}-CC$)')

    if correct_energy[0] is True or modelname == 'shallow_reco':
        ax.set_ylabel('Corrected reconstructed energy (GeV)')

    cbar2 = fig.colorbar(energy_res_shower, ax=ax)
    cbar2.ax.set_ylabel('Number of events')

    pdf_plots.savefig(fig)
    pdf_plots.close()


def get_boolean_track_and_shower_separation(particle_type, is_cc):
    """

    :param particle_type:
    :param is_cc:
    :return:
    """
    # Input: np arrays from mc_info
    # Output: np arrays type bool which identify track/shower like events

    # track: muon/a-muon CC --> 14, True
    # shower: elec/a-elec; muon/a-muon NC
    # particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
    # In my dataset, there are actually no NC events

    abs_particle_type = np.abs(particle_type)
    track = np.logical_and(abs_particle_type == 14, is_cc == True)
    shower = np.logical_or(np.logical_and(abs_particle_type == 14, is_cc == False),
                           abs_particle_type == 12)

    return track, shower


def correct_reco_energy(arr_nn_pred, metric='median'):
    """

    :param arr_nn_pred:
    :param metric:
    :return:
    """
    is_track, is_shower = get_boolean_track_and_shower_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3])

    energy_mc = arr_nn_pred[:, 4][is_shower]
    energy_pred = arr_nn_pred[:, 9][is_shower]

    arr_nn_pred_corr = np.copy(arr_nn_pred)

    correction_factors_x = []
    correction_factors_y = []

    e_range = np.logspace(np.log(3)/np.log(2),np.log(100)/np.log(2),50,base=2)
    n_ranges = e_range.shape[0] - 1
    for i in xrange(n_ranges):
        e_range_low, e_range_high = e_range[i], e_range[i+1]
        e_range_mean = (e_range_high + e_range_low) / float(2)

        e_mc_cut_boolean = np.logical_and(e_range_low < energy_mc, energy_mc <= e_range_high)
        e_mc_cut = energy_mc[e_mc_cut_boolean]
        e_pred_cut = energy_pred[e_mc_cut_boolean]

        if metric == 'median':
            correction_factor = np.median((e_pred_cut - e_mc_cut) / e_pred_cut)
        elif metric == 'mean':
            correction_factor = np.mean((e_pred_cut - e_mc_cut) / e_pred_cut)
        else:
            raise ValueError('The specified metric "' + metric + '" is not implemented.')

        correction_factors_x.append(e_range_mean)
        correction_factors_y.append(correction_factor)

    # linear interpolation of correction factors

    correction_factor_en_pred = np.interp(energy_pred, correction_factors_x, correction_factors_y)
    arr_nn_pred_corr[:, 9][is_shower] = energy_pred + (- correction_factor_en_pred) * energy_pred # apply correction

    return arr_nn_pred_corr


def make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median_relative', energy_bins=np.linspace(3,100,32),
                                              precuts=(False, '3-100_GeV_prod'), correct_energy=(True, 'median'), compare_shallow=(False, None)):
    """

    :param arr_nn_pred:
    :param modelname:
    :param metric:
    :param energy_bins:
    :param precuts:
    :param correct_energy:
    :param compare_shallow:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])
    if correct_energy[0] is True:
        arr_nn_pred = correct_reco_energy(arr_nn_pred, metric=correct_energy[1])

    if compare_shallow[0] is True:
        arr_nn_pred_shallow = compare_shallow[1]
        if precuts[0] is True:
            arr_nn_pred_shallow = arr_nn_pred_select_pheid_events(arr_nn_pred_shallow, invert=False, precuts=precuts[1])
        energy_metric_plot_data_shallow = calculate_plot_data_of_energy_dependent_label(
        arr_nn_pred_shallow, energy_bins=energy_bins, label=('energy', metric))

    energy_metric_plot_data = calculate_plot_data_of_energy_dependent_label(arr_nn_pred, energy_bins=energy_bins, label=('energy', metric))
    energy_metric_plot_data_shower, energy_metric_plot_data_track = energy_metric_plot_data[0], energy_metric_plot_data[1]

    fig, ax = plt.subplots()

    bins = energy_metric_plot_data_track[0]
    metric_track = energy_metric_plot_data_track[1]
    metric_shower = energy_metric_plot_data_shower[1]

    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/1d/energy/energy_resolution_' + modelname + '.pdf')

    ax.step(bins, metric_track, linestyle="-", where='post', label='OrcaNet')
    if compare_shallow[0] is True: ax.step(bins, energy_metric_plot_data_shallow[1][1], linestyle="-", where='post', label='Standard Reco')

    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
    x_ticks_major = np.arange(0, 101, 10)
    ax.set_xticks(x_ticks_major)
    ax.minorticks_on()
    title = plt.title(reco_name + 'Track like (' + r'$\nu_{\mu}-CC$)')
    title.set_position([.5, 1.04])
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median relative error (energy)')
    ax.grid(True)
    ax.legend(loc='upper right')

    pdf_plots.savefig(fig)
    ax.cla()


    ax.step(bins, metric_shower, linestyle="-", where='post', label='OrcaNet')
    if compare_shallow[0] is True: ax.step(bins, energy_metric_plot_data_shallow[0][1], linestyle="-", where='post', label='Standard Reco')

    ax.set_xticks(x_ticks_major)
    ax.minorticks_on()
    title = plt.title(reco_name + 'Shower like (' + r'$\nu_{e}-CC$)')
    title.set_position([.5, 1.04])
    corr = 'corrected ' if correct_energy[0] is True else ''
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median relative error (' + corr + 'energy)')
    ax.grid(True)
    ax.legend(loc='upper right')

    pdf_plots.savefig(fig)

    plt.close()
    pdf_plots.close()


def calculate_plot_data_of_energy_dependent_label(arr_nn_pred, energy_bins=np.linspace(3,100,20), label=('energy', 'median_relative')):
    """
    Generate binned statistics for the energy mae, or the relative mae. Separately for track and shower events.
    :param arr_nn_pred:
    :param energy_bins:
    :param label:
    :return:
    """
    labels = {'energy': {'reco_index': 9, 'mc_index': 4}, 'dir_x': {'reco_index': 10, 'mc_index':6},
              'dir_y': {'reco_index': 11, 'mc_index':7}, 'dir_z': {'reco_index': 12, 'mc_index':8},
              'bjorken_y': {'reco_index': 13, 'mc_index':5}}
    label_name, metric = label[0], label[1]
    mc_label = arr_nn_pred[:, labels[label_name]['mc_index']]
    reco_label = arr_nn_pred[:, labels[label_name]['reco_index']] # reconstruction results for the chosen label

    is_track, is_shower = get_boolean_track_and_shower_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3])

    if label_name == 'energy' or label_name == 'bjorken_y':
        err = np.abs(reco_label - mc_label)
    elif label_name in ['dir_x', 'dir_y', 'dir_z']:
        reco_label_list = []
        for i in xrange(reco_label.shape[0]):
            dir = reco_label[i]
            if dir < -1: dir = -1
            if dir > 1: dir = 1
            reco_label_list.append(dir)

        reco_label = np.array(reco_label_list)

        #err = np.abs(np.arccos(-reco_label), - np.arccos(-mc_label))
        err = np.abs(reco_label - mc_label)
    else:
        raise ValueError('The label' + str(label[0]) + ' is not available.')

    mc_energy = arr_nn_pred[:, 4]
    energy_to_label_performance_plot_data_shower = bin_error_in_energy_bins(energy_bins, mc_energy[is_shower], err[is_shower], operation=metric)
    energy_to_label_performance_plot_data_track = bin_error_in_energy_bins(energy_bins, mc_energy[is_track], err[is_track], operation=metric)
    energy_to_label_performance_plot_data = [energy_to_label_performance_plot_data_shower, energy_to_label_performance_plot_data_track]

    return energy_to_label_performance_plot_data


def bin_error_in_energy_bins(energy_bins, mc_energy, err, operation='median_relative'):
    """

    :param energy_bins:
    :param mc_energy:
    :param err:
    :param operation:
    :return:
    """
    # bin the err, depending on their mc_energy, into energy_bins
    hist_energy_losses = np.zeros((len(energy_bins) - 1))
    hist_energy_variance = np.zeros((len(energy_bins) - 1))
    # In which bin does each event belong, according to its mc energy:
    bin_indices = np.digitize(mc_energy, bins=energy_bins)

    # For every mc energy bin, calculate the merge operation (e.g. mae or median relative) of all events that have a corresponding mc energy
    for bin_no in xrange(1, len(energy_bins)):
        current_err = err[bin_indices == bin_no]
        current_mc_energy = mc_energy[bin_indices == bin_no]

        if operation == 'mae':
            # calculate mean absolute error
            hist_energy_losses[bin_no - 1] = np.mean(current_err)
        elif operation == 'median_relative':
            # calculate the median of the relative error: |label_reco-label_true|/E_true and also its variance
            # probably makes only sense if the label from the err array is the energy
            relative_error = current_err / current_mc_energy
            hist_energy_losses[bin_no - 1] = np.median(relative_error)
            hist_energy_variance[bin_no - 1] = np.var(relative_error)
        elif operation == 'mean_relative':
            # calculate the median of the relative error: |label_reco-label_true|/E_true and also its variance
            # probably makes only sense if the label from the err array is the energy
            relative_error = current_err / current_mc_energy
            hist_energy_losses[bin_no - 1] = np.mean(relative_error)
            hist_energy_variance[bin_no - 1] = np.var(relative_error)
        elif operation == 'median':
            hist_energy_losses[bin_no - 1] = np.median(current_err)
        else:
            raise ValueError('Operation modes other than "mae" and "median_relative" are not supported.')

    # For proper plotting with plt.step where="post"
    hist_energy_losses = np.append(hist_energy_losses, hist_energy_losses[-1])
    hist_energy_variance = np.append(hist_energy_variance, hist_energy_variance[-1]) if operation == 'median_relative' else None
    energy_binned_err_plot_data = [energy_bins, hist_energy_losses, hist_energy_variance]

    return energy_binned_err_plot_data


def make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, energy_bins=np.linspace(3,100,49), precuts=(False, '3-100_GeV_prod'),
                                       compare_shallow=(False, None), correct_energy=(False, 'median')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param energy_bins:
    :param precuts:
    :param compare_shallow:
    :param correct_energy:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])
    if correct_energy[0] is True:
        arr_nn_pred = correct_reco_energy(arr_nn_pred, metric=correct_energy[1])

    std_rel_track, std_rel_shower = get_std_rel_plot_data(arr_nn_pred, energy_bins)

    if compare_shallow[0] is True:
        arr_nn_pred_shallow = compare_shallow[1]
        if precuts[0] is True:
            arr_nn_pred_shallow = arr_nn_pred_select_pheid_events(arr_nn_pred_shallow, invert=False, precuts=precuts[1])
        std_rel_track_shallow, std_rel_shower_shallow = get_std_rel_plot_data(arr_nn_pred_shallow, energy_bins)
        print std_rel_shower_shallow

    fig, ax = plt.subplots()

    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/1d/energy/energy_std_rel_' + modelname + '.pdf')

    ax.step(energy_bins, std_rel_track, linestyle="-", where='post', label='OrcaNet')
    if compare_shallow[0] is True: ax.step(energy_bins, std_rel_track_shallow, linestyle="-", where='post', label='Standard Reco')

    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
    x_ticks_major = np.arange(0, 101, 10)
    ax.set_xticks(x_ticks_major)
    ax.minorticks_on()
    title = plt.title(reco_name + 'Track like (' + r'$\nu_{\mu}-CC$)')
    title.set_position([.5, 1.04])
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel(r'$\sigma / E_{true}$')
    ax.grid(True)

    ax.legend(loc='upper right')
    pdf_plots.savefig(fig)
    ax.cla()

    ax.step(energy_bins, std_rel_shower, linestyle="-", where='post', label='OrcaNet')
    if compare_shallow[0] is True: ax.step(energy_bins, std_rel_shower_shallow, linestyle="-", where='post', label='Standard Reco')

    ax.set_xticks(x_ticks_major)
    ax.minorticks_on()
    title = plt.title(reco_name + 'Shower like (' + r'$\nu_{e}-CC$)')
    title.set_position([.5, 1.04])
    corr = ' (corrected energy)' if correct_energy[0] is True else ''
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel(r'$\sigma / E_{true}$' + corr)
    ax.grid(True)

    ax.legend(loc='upper right')
    pdf_plots.savefig(fig)

    plt.close()
    pdf_plots.close()


def get_std_rel_plot_data(arr_nn_pred, energy_bins):
    """

    :param arr_nn_pred:
    :param energy_bins:
    :return:
    """
    energy_mc = arr_nn_pred[:, 4]
    energy_pred = arr_nn_pred[:, 9]
    print np.amax(energy_mc)
    print np.amin(energy_mc)
    print np.amax(energy_pred)
    print np.amin(energy_pred)
    is_track, is_shower = get_boolean_track_and_shower_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3])

    energy_mc_track, energy_mc_shower = energy_mc[is_track], energy_mc[is_shower]
    energy_pred_track, energy_pred_shower = energy_pred[is_track], energy_pred[is_shower]

    std_rel_track, std_rel_shower = [], [] # y-axis of the plot
    for i in xrange(energy_bins.shape[0] -1):
        e_range_low, e_range_high = energy_bins[i], energy_bins[i+1]
        e_range_mean = (e_range_low + e_range_high)/ float(2)

        e_pred_track_cut_boolean = np.logical_and(e_range_low < energy_mc_track, energy_mc_track <= e_range_high)
        e_pred_shower_cut_boolean = np.logical_and(e_range_low < energy_mc_shower, energy_mc_shower <= e_range_high)
        e_pred_track_cut = energy_pred_track[e_pred_track_cut_boolean]
        e_pred_shower_cut = energy_pred_shower[e_pred_shower_cut_boolean]

        std_track_temp, std_shower_temp = np.std(e_pred_track_cut), np.std(e_pred_shower_cut)
        std_rel_track.append(std_track_temp / float(e_range_mean))
        std_rel_shower.append(std_shower_temp / float(e_range_mean))

    # fix for mpl
    std_rel_track.append(std_rel_track[-1]), std_rel_shower.append(std_rel_shower[-1])

    return std_rel_track, std_rel_shower


def make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median', energy_bins=np.linspace(3,100,32),
                                      precuts=(False, '3-100_GeV_prod'), compare_shallow=(False, None)):
    """

    :param arr_nn_pred:
    :param modelname:
    :param metric:
    :param energy_bins:
    :param precuts:
    :param compare_shallow:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])
    if compare_shallow[0] is True:
        arr_nn_pred_shallow = compare_shallow[1]
        if precuts[0] is True:
            arr_nn_pred_shallow = arr_nn_pred_select_pheid_events(arr_nn_pred_shallow, invert=False, precuts=precuts[1])

    directions = ['dir_x', 'dir_y', 'dir_z']

    fig, ax = plt.subplots()

    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/1d/dir/dir_resolution_' + modelname + '.pdf')
    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '

    for i, direction in enumerate(directions):

        dir_plot_data = calculate_plot_data_of_energy_dependent_label(arr_nn_pred, energy_bins=energy_bins, label=(direction, metric))
        dir_plot_data_shower, dir_plot_data_track = dir_plot_data[0], dir_plot_data[1]
        if compare_shallow[0] is True:
            dir_plot_data_shallow = calculate_plot_data_of_energy_dependent_label(arr_nn_pred_shallow, energy_bins=energy_bins, label=(direction, metric))

        bins = dir_plot_data_track[0]
        dir_perf_track = dir_plot_data_track[1]

        ax.step(bins, dir_perf_track, linestyle="-", where='post', label='DL ' + direction) # track
        if compare_shallow[0] is True: ax.step(bins, dir_plot_data_shallow[1][1], linestyle="-", where='post', label='Std ' + direction)

        x_ticks_major = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks_major)
        ax.minorticks_on()
        title = plt.title(reco_name + 'Track like (' + r'$\nu_{\mu}-CC$)')
        title.set_position([.5, 1.04])
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median error dir')
        ax.grid(True)
        ax.legend(loc='upper right')

    pdf_plots.savefig(fig)
    ax.cla()

    for i, direction in enumerate(directions):
        dir_plot_data = calculate_plot_data_of_energy_dependent_label(arr_nn_pred,energy_bins=energy_bins,label=(direction, metric))
        dir_plot_data_shower, dir_plot_data_track = dir_plot_data[0], dir_plot_data[1]
        if compare_shallow[0] is True:
            dir_plot_data_shallow = calculate_plot_data_of_energy_dependent_label(arr_nn_pred_shallow, energy_bins=energy_bins, label=(direction, metric))

        bins = dir_plot_data_track[0]
        dir_perf_shower = dir_plot_data_shower[1]

        ax.step(bins, dir_perf_shower, linestyle="-", where='post', label='DL ' + direction) # shower
        if compare_shallow[0] is True: ax.step(bins, dir_plot_data_shallow[0][1], linestyle="-", where='post', label='Std ' + direction)

        ax.set_xticks(x_ticks_major)
        ax.minorticks_on()
        title = plt.title(reco_name + 'Shower like (' + r'$\nu_{e}-CC$)')
        title.set_position([.5, 1.04])
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median error dir')
        ax.grid(True)
        ax.legend(loc='upper right')

    pdf_plots.savefig(fig)

    plt.close()
    pdf_plots.close()


def make_2d_dir_correlation_plot(arr_nn_pred, modelname, dir_bins=np.linspace(-1,1,100), precuts=(False, '3-100_GeV_prod')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param dir_bins:
    :param precuts:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])

    labels = {'dir_x': {'reco_index': 10, 'mc_index':6}, 'dir_y': {'reco_index': 11, 'mc_index':7},
              'dir_z': {'reco_index': 12, 'mc_index':8}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/2d/dir/dir_correlation_' + modelname + '.pdf')

    is_track, is_shower = get_boolean_track_and_shower_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3])

    plot_2d_dir_correlation(arr_nn_pred, is_track, is_shower, 'dir_x', labels, dir_bins, fig, ax, pdf_plots, modelname)
    plot_2d_dir_correlation(arr_nn_pred, is_track, is_shower, 'dir_y', labels, dir_bins, fig, ax, pdf_plots, modelname)
    plot_2d_dir_correlation(arr_nn_pred, is_track, is_shower, 'dir_z', labels, dir_bins, fig, ax, pdf_plots, modelname)

    plt.close()
    pdf_plots.close()


#-Regression -#

def plot_2d_dir_correlation(arr_nn_pred, is_track, is_shower, label, labels, dir_bins, fig, ax, pdf_plots, modelname):
    """

    :param arr_nn_pred:
    :param is_track:
    :param is_shower:
    :param label:
    :param labels:
    :param dir_bins:
    :param fig:
    :param ax:
    :param pdf_plots:
    :param modelname:
    :return:
    """
    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '

    dir_mc = arr_nn_pred[:, labels[label]['mc_index']]
    dir_pred = arr_nn_pred[:, labels[label]['reco_index']]

    hist_2d_dir_shower = np.histogram2d(dir_mc[is_shower], dir_pred[is_shower], dir_bins)
    hist_2d_dir_track = np.histogram2d(dir_mc[is_track], dir_pred[is_track], dir_bins)
    bin_edges_dir = hist_2d_dir_shower[1] # doesn't matter if we take shower/track and x/y, bin edges are same for all of them

    dir_corr_track = ax.pcolormesh(bin_edges_dir, bin_edges_dir, hist_2d_dir_track[0].T,
                                     norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_dir_track[0].T.max()))

    title = plt.title(reco_name + 'Track like events (' + r'$\nu_{\mu}-CC$)')
    title.set_position([.5, 1.04])
    cbar1 = fig.colorbar(dir_corr_track, ax=ax)
    cbar1.ax.set_ylabel('Number of events')
    ax.set_xlabel('True direction [' + label + ']'), ax.set_ylabel('Reconstructed direction [' + label + ']')
    plt.tight_layout()

    pdf_plots.savefig(fig)
    cbar1.remove()
    dir_corr_track.remove()

    dir_corr_shower = ax.pcolormesh(bin_edges_dir, bin_edges_dir, hist_2d_dir_shower[0].T,
                                    norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_dir_shower[0].T.max()))
    plt.title(reco_name + 'Shower like events (' + r'$\nu_{e}-CC$)')

    cbar2 = fig.colorbar(dir_corr_shower, ax=ax)
    cbar2.ax.set_ylabel('Number of events')

    pdf_plots.savefig(fig)
    cbar2.remove()
    plt.cla()


def make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median', energy_bins=np.linspace(3,100,32),
                                            precuts=(False, '3-100_GeV_prod'), compare_shallow=(False, None)):
    """

    :param arr_nn_pred:
    :param modelname:
    :param metric:
    :param energy_bins:
    :param precuts:
    :param compare_shallow:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])
    if compare_shallow[0] is True:
        arr_nn_pred_shallow = compare_shallow[1]
        if precuts[0] is True:
            arr_nn_pred_shallow = arr_nn_pred_select_pheid_events(arr_nn_pred_shallow, invert=False, precuts=precuts[1])
        bjorken_y_metric_plot_data_shallow = calculate_plot_data_of_energy_dependent_label(arr_nn_pred_shallow, energy_bins=energy_bins,
                                                                                           label=('bjorken_y', metric))

    bjorken_y_metric_plot_data = calculate_plot_data_of_energy_dependent_label(arr_nn_pred, energy_bins=energy_bins, label=('bjorken_y', metric))
    bjorken_y_metric_plot_data_shower, bjorken_y_metric_plot_data_track = bjorken_y_metric_plot_data[0], bjorken_y_metric_plot_data[1]

    fig, ax = plt.subplots()

    bins = bjorken_y_metric_plot_data_track[0]
    metric_track = bjorken_y_metric_plot_data_track[1]
    metric_shower = bjorken_y_metric_plot_data_shower[1]

    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/1d/bjorken_y/bjorken_y_' + modelname + '.pdf')

    ax.step(bins, metric_track, linestyle="-", where='post', label='OrcaNet') # track
    if compare_shallow[0] is True: ax.step(bins, bjorken_y_metric_plot_data_shallow[1][1], linestyle="-", where='post', label='Standard Reco')

    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
    x_ticks_major = np.arange(0, 101, 10)
    ax.set_xticks(x_ticks_major)
    ax.minorticks_on()
    title = plt.title(reco_name + 'Track like (' + r'$\nu_{\mu}-CC$)')
    title.set_position([.5, 1.04])
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median error bjorken-y')
    ax.grid(True)
    ax.legend(loc='upper right')

    pdf_plots.savefig(fig)
    ax.cla()

    ax.step(bins, metric_shower, linestyle="-", where='post', label='OrcaNet') # shower
    if compare_shallow[0] is True: ax.step(bins, bjorken_y_metric_plot_data_shallow[0][1], linestyle="-", where='post', label='Standard Reco')

    ax.set_xticks(x_ticks_major)
    ax.minorticks_on()
    title = plt.title(reco_name + 'Shower like (' + r'$\nu_{e}-CC$)')
    title.set_position([.5, 1.04])
    ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median error bjorken-y')
    ax.grid(True)
    ax.legend(loc='upper right')

    pdf_plots.savefig(fig)

    plt.close()
    pdf_plots.close()



#------------- Functions used in making Matplotlib plots -------------#








