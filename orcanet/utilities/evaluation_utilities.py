#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code for the evaluation of a network's performance after training."""

import os
import math
import h5py
import numpy as np
import keras as ks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from orcanet.utilities.nn_utilities import generate_batches_from_hdf5_file

# ------------- Functions used in evaluating the performance of model -------------#


def get_nn_predictions_and_mc_info(cfg, model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, samples=None):
    """
    Creates an arr_nn_pred array based on the test_data.
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
    :return ndarray arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
    """
    # get total number of samples
    cum_number_of_steps = get_cum_number_of_steps(test_files, batchsize)
    ax = np.newaxis

    arr_nn_pred, samples_f = None, None
    for f_number, f in enumerate(test_files):

        if samples is None:
            test_file = h5py.File(f[0][0], 'r')
            samples_f = test_file['y'].shape[0]
            test_file.close()
        else:
            samples_f = samples
        steps = int(samples_f/batchsize)

        generator = generate_batches_from_hdf5_file(cfg, f[0], zero_center_image=xs_mean, yield_mc_info=True)

        arr_nn_pred_row_start = cum_number_of_steps[f_number] * batchsize
        for s in range(steps):
            if s % 100 == 0: print('Predicting in step ' + str(s) + ' on file ' + str(f_number))

            xs, y_true, mc_info = next(generator)
            y_pred = model.predict_on_batch(xs)

            if class_type[1] == 'energy_and_direction_and_bjorken-y':
                # TODO temp old 60b prod
                y_pred = np.concatenate([y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4]], axis=1)
                y_true = np.concatenate([y_true['energy'], y_true['dir_x'], y_true['dir_y'], y_true['dir_z'], y_true['bjorken-y']], axis=1) # dont need to save y_true err input
            elif class_type[1] == 'energy_dir_bjorken-y_errors':
                y_pred = np.concatenate(y_pred, axis=1)
                y_true = np.concatenate([y_true['e'], y_true['dir_x'], y_true['dir_y'], y_true['dir_z'], y_true['by']], axis=1) # dont need to save y_true err input
            elif class_type[1] == 'energy_dir_bjorken-y_vtx_errors':
                y_pred = np.concatenate(y_pred, axis=1)
                y_true = np.concatenate([y_true['e'], y_true['dx'], y_true['dy'], y_true['dz'], y_true['by'], y_true['vx'], y_true['vy'], y_true['vz'], y_true['vt']],
                                        axis=1)  # dont need to save y_true err input
            else:
                raise NameError("Unknown class_type " + str(class_type[1]))
            # mc labels
            energy = mc_info[:, 2]
            particle_type = mc_info[:, 1]
            is_cc = mc_info[:, 3]
            event_id = mc_info[:, 0]
            run_id = mc_info[:, 9]
            bjorken_y = mc_info[:, 4]
            dir_x, dir_y, dir_z = mc_info[:, 5], mc_info[:, 6], mc_info[:, 7]
            vtx_x, vtx_y, vtx_z = mc_info[:, 10], mc_info[:, 11], mc_info[:, 12]
            time_residual_vtx = mc_info[:, 13]
            # if mc_info.shape[1] > 13: TODO add prod_ident

            # make a temporary energy_correct array for this batch
            # arr_nn_pred_temp = np.concatenate([run_id[:, ax], event_id[:, ax], particle_type[:, ax], is_cc[:, ax], energy[:, ax],
            #                                    bjorken_y[:, ax], dir_x[:, ax], dir_y[:, ax], dir_z[:, ax], y_pred, y_true], axis=1)
            #print(y_pred.shape)
            #print(y_true.shape)
            # alle [...,ax] haben dim 64,1.
            arr_nn_pred_temp = np.concatenate([run_id[:, ax], event_id[:, ax], particle_type[:, ax], is_cc[:, ax],
                                               energy[:, ax], bjorken_y[:, ax], dir_x[:, ax], dir_y[:, ax],
                                               dir_z[:, ax], vtx_x[:, ax], vtx_y[:, ax], vtx_z[:, ax],
                                               time_residual_vtx[:, ax], y_pred, y_true], axis=1)

            if arr_nn_pred is None: arr_nn_pred = np.zeros((cum_number_of_steps[-1] * batchsize, arr_nn_pred_temp.shape[1:2][0]), dtype=np.float32)
            arr_nn_pred[arr_nn_pred_row_start + s*batchsize : arr_nn_pred_row_start + (s+1) * batchsize] = arr_nn_pred_temp

    # make_pred_h5_file(arr_nn_pred, filepath='predictions/' + modelname, mc_prod='3-100GeV')

    return arr_nn_pred


def get_array_indices():

    label_indices = {'run_id': 0, 'event_id': 1, 'particle_type': 2, 'is_cc': 3, 'bjorken_y': {'true_index': 5, 'reco_index': 17},
                     'energy': {'true_index': 4, 'reco_index': 13, 'index_label_std': 19, 'str': ('energy', 'GeV')},
                     'dir_x': {'true_index':6, 'reco_index': 14, 'reco_index_std': 21, 'str': ('dir_x', 'rad')},
                     'dir_y': {'true_index':7, 'reco_index': 15, 'reco_index_std': 23, 'str': ('dir_y', 'rad')},
                     'dir_z': {'true_index':8, 'reco_index': 16, 'reco_index_std': 25, 'str': ('dir_z', 'rad')},
                     'vtx_x': {'true_index': 9},
                     'vtx_y': {'true_index': 10},
                     'vtx_z': {'true_index': 11},
                     'time_residual_vtx': {'true_index': 12}}

    return label_indices


def get_cum_number_of_steps(files, batchsize):
    """
    Function that calculates the cumulative number of prediction steps for the single files in the <files> list.
    Typically used during prediction when the data is split to multiple input files.
    :param list(tuple(list,int)) files: file list that should have the shape [ ( [], ), ... ]
    :param int batchsize: batchsize that is used during the prediction
    """
    cum_number_of_steps = [0]
    for i, f in enumerate(files):
        samples = h5py.File(f[0][0], 'r')['y'].shape[0]
        steps = int(samples/batchsize)
        cum_number_of_steps.append(cum_number_of_steps[i] + steps) # [0, steps_sample_1, steps_sample_1 + steps_sample_2, ...]

    return cum_number_of_steps


def make_pred_h5_file(arr_nn_pred, filepath, mc_prod='3-100GeV'):
    """
    Takes an arr_nn_pred for the track/shower classification and saves the important reco columns to a .h5 file.
    :param ndarray arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
    :param str mc_prod: optional parameter that specifies which mc prod is used. E.g. 3-100GeV or 1-5GeV.
    :param str filepath: filepath that should be used for saving the .h5 file.
    """
    run_id = arr_nn_pred[:, 0:1]
    event_id = arr_nn_pred[:, 1:2]
    particle_type = arr_nn_pred[:, 2:3]
    is_cc = arr_nn_pred[:, 3:4]
    y_pred_track = arr_nn_pred[:, 9:10]

    arr_nn_output = np.concatenate([run_id, event_id, particle_type, is_cc, y_pred_track], axis=1)

    f = h5py.File(filepath + '_' + mc_prod + '.h5', 'w')
    dset = f.create_dataset('nn_output', data=arr_nn_output)
    dset.attrs['array_contents'] = 'Columns: run_id, event_id, PID (particle_type + is_cc), y_pred_track. ' \
                                   'PID info: (12, 0): elec-NC, (12, 1): elec-CC, (14, 1): muon-CC, (16, 1): tau-CC. ' \
                                   'y_pred_track info: probability of the neural network for this event to be a track.'
    f.close()
    #arr_nn_output = np.rec.fromarrays([run_id, event_id, particle_type, is_cc, y_pred_track],
    #                                  dtype=[('run_id','f4'),('event_id','f4'),('particle_type','f4'),('is_cc','f4'),
    #                                  ('y_pred_track','f4')])

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
    Loads the pheid events that survive the precuts from a .txt file, adds a pid column to them and returns it.
    :param str precuts: specifies, which precuts should be loaded.
    :return ndarray(ndim=2) arr_pheid_sel_events: 2D array that contains [particle_type, is_cc, event_id, run_id]
                                                   for each event that survives the precuts.
    """
    path = 'results/plots/pheid_event_selection_txt/' # folder for storing the precut .txts

    #### Precuts
    if precuts == '3-100_GeV_prod':
        particle_type_dict = {'muon-CC': [['muon_cc_3_100_selectedShifted.txt'], (14,1)],
                              'elec-CC': [['elec_cc_3_100_selectedShifted.txt'], (12,1)],
                              'elec-NC': [['elec_nc_3_100_selectedShifted.txt'], (12, 0)],
                              'tau-CC': [['tau_cc_3_100_selectedShifted.txt'], (16, 1)]}

    elif precuts == '1-5_GeV_prod':
        particle_type_dict = {'muon-CC': [['muon_cc_1_5_selectedShifted.txt'], (14,1)],
                              'elec-CC': [['elec_cc_1_5_selectedShifted.txt'], (12,1)],
                              'elec-NC': [['elec_nc_1_5_selectedShifted.txt'], (12, 0)]}

    elif precuts == '3-100_GeV_containment_cut':
        particle_type_dict = {'muon-CC': [['old/muon_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt'], (14,1)],
                              'elec-CC': [['old/elec_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt'], (12,1)]}

    elif precuts == '3-100_GeV_prod_energy_comparison':
        path = '/home/woody/capn/mppi033h/Data/various/cuts_txt_files/'
        particle_type_dict = {'muon-CC': [['cuts_shallow_3_100_muon_cc.txt'], (14,1)],
                              'elec-CC': [['cuts_shallow_3_100_elec_cc.txt'], (12,1)]}

    elif precuts == '3-100_GeV_prod_energy_comparison_old_evt_id':
        path = '/home/woody/capn/mppi033h/Data/various/cuts_txt_files/'
        particle_type_dict = {'muon-CC': [['cuts_shallow_3_100_muon_cc_old_evt_id.txt'], (14,1)],
                              'elec-CC': [['cuts_shallow_3_100_elec_cc_old_evt_id.txt'], (12,1)]}

    elif precuts == '3-100_GeV_prod_energy_comparison_is_good':
        path = '/home/woody/capn/mppi033h/Data/various/cuts_txt_files/'
        particle_type_dict = {'muon-CC': [['cuts_shallow_3_100_muon_cc_is_good.txt'], (14,1)],
                              'elec-CC': [['cuts_shallow_3_100_elec_cc_is_good.txt'], (12,1)]}

    elif precuts == 'regr_3-100_GeV_prod_and_1-3_GeV_prod':
        path = '/home/woody/capn/mppi033h/Data/various/cuts_txt_files/'
        particle_type_dict = {'muon-CC': [['cuts_shallow_3_100_muon-cc.txt', 'cuts_shallow_1_3_muon-cc.txt'], (14,1)],
                              'elec-CC': [['cuts_shallow_3_100_elec-cc.txt', 'cuts_shallow_1_3_elec-cc.txt'], (12,1)],
                              'elec-NC': [['cuts_shallow_3_100_elec-nc.txt', 'cuts_shallow_1_3_elec-nc.txt'], (12,0)],
                              'tau-CC': [['cuts_shallow_3_100_tau-cc.txt'], (16,1)]}

    else:
        raise ValueError('The specified precuts option "' + str(precuts) + '" is not available.')

    arr_pheid_sel_events = None
    for key in particle_type_dict:
        for i, txt_file in enumerate(particle_type_dict[key][0]):

            if arr_pheid_sel_events is None:
                arr_pheid_sel_events = np.loadtxt(path + txt_file, dtype=np.float32)
                arr_pheid_sel_events = add_pid_column_to_array(arr_pheid_sel_events, particle_type_dict, key)

                # add prod col, i=0 for 3-100GeV and i=1 for 1-5 GeV
                if precuts == '1-5_GeV_prod': i = 1 # in the case of PID with these precuts, TODO fix

                arr_pheid_sel_events = np.concatenate([arr_pheid_sel_events, np.full((arr_pheid_sel_events.shape[0], 1), i, dtype=np.float32)], axis=1)

            else:
                temp_pheid_sel_events = np.loadtxt(path + txt_file, dtype=np.float32)
                temp_pheid_sel_events = add_pid_column_to_array(temp_pheid_sel_events, particle_type_dict, key)

                # add prod col, i=0 for 3-100GeV and i=1 for 1-5 GeV
                if precuts == '1-5_GeV_prod': i = 1

                temp_pheid_sel_events = np.concatenate([temp_pheid_sel_events, np.full((temp_pheid_sel_events.shape[0], 1), i, dtype=np.float32)], axis=1)

                arr_pheid_sel_events = np.concatenate((arr_pheid_sel_events, temp_pheid_sel_events), axis=0)

    return arr_pheid_sel_events


def add_prod_column_to_cut_arr_nn_pred(cut_arr_nn_pred, arr_nn_pred_e_col):
    """
    Important: don't change the order (axis_0) for the cut_arr_nn_pred!
    :param cut_arr_nn_pred:
    :param arr_nn_pred_e_col:
    :return:
    """
    # add new column with placeholder zeros to the array
    zeros = np.zeros((cut_arr_nn_pred.shape[0], 1), dtype=np.float32)
    cut_arr_nn_pred = np.concatenate([cut_arr_nn_pred, zeros], axis=1) # 0,1,2,3,4: run_id, event_id, particle_type, is_cc, zeros

    # yields array with 0 = > 3 GeV (3-100 GeV prod) and 1 = < 3 GeV (1-5 GeV prod)
    boolean_is_low_e_prod = arr_nn_pred_e_col < 3
    cut_arr_nn_pred[:, 4] = boolean_is_low_e_prod.astype(np.float32) # index 4 is the last column

    return cut_arr_nn_pred # now with prod col


def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed on the entire row.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def in_nd(a, b, assume_unique=False):
    """
    Function that generalizes the np in_1d function to nd.
    Checks if entries in axis_0 of a exist in b and returns the bool array for all rows.
    Kind of hacky by using str views on the np arrays.
    :param ndarray(ndim=2) a: array where it should be checked whether each row exists in b or not.
    :param ndarray(ndim=2) b: array upon which the rows of a are checked.
    :param bool assume_unique: if True, the input arrays are both assumed to be unique, which can speed up the calculation.
    :return: ndarray(ndim=1): Boolean array that specifies for each row of a if it also exists in b or not.

    """
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)

def test_ind():
    a = np.array([[1,129385,1,-1,-0],[1,1,0,0,0],[0,0,0,0,0]])
    b = np.array([[1,1,0,0,2],[6,6,6,6,6],[1,129385,1,-1,-0]])

    c = in_nd(a,b)
    it_works = np.all(c == [ True, False, False ])
    return it_works


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
    arr_sel_events = load_pheid_event_selection(precuts=precuts)
    cut_arr_nn_pred = arr_nn_pred[:, [0, 1, 2, 3]] # 0,1,2,3: run_id, event_id, particle_type, is_cc
    cut_arr_nn_pred = add_prod_column_to_cut_arr_nn_pred(cut_arr_nn_pred, arr_nn_pred[:, 4])

    bool_evt_run_id_in_selection = in_nd(cut_arr_nn_pred, arr_sel_events)

    if invert is True: bool_evt_run_id_in_selection = np.invert(bool_evt_run_id_in_selection)

    arr_nn_pred = arr_nn_pred[bool_evt_run_id_in_selection] # apply boolean in_pheid selection to the array

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
    print(print_text, '\n', str(performance *100), '\n', arr_energy_correct.shape)

#-- Utility functions --#

#-- Functions for making energy to accuracy plots --#

#- Classification -#

def make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title, filename, plot_range=(1, 100), precuts=(False, '3-100_GeV_prod'), corr_cut_pred_0=0.5):
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

    hist_1d_energy_class = np.histogram(energy_class, bins=99, range=plot_range) # TODO standard 97
    hist_1d_energy_correct_class = np.histogram(arr_nn_pred_class[correct_class == 1, 4], bins=99, range=plot_range) # TODO, 97

    bin_edges = hist_1d_energy_class[1]
    hist_1d_energy_accuracy_class_bins = np.divide(hist_1d_energy_correct_class[0], hist_1d_energy_class[0], dtype=np.float32) # TODO solve division by zero

    if invert is True: hist_1d_energy_accuracy_class_bins = np.absolute(hist_1d_energy_accuracy_class_bins - 1)

    # For making it work with matplotlib step plot
    hist_1d_energy_accuracy_class_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_class_bins))

    label = {'muon-CC': r'$\nu_{\mu}-CC$', 'a_muon-CC': r'$\overline{\nu}_{\mu}-CC$', 'elec-CC': r'$\nu_{e}-CC$', 'a_elec-CC': r'$\overline{\nu}_{e}-CC$',
             'elec-NC': r'$\nu_{e}-NC$', 'a_elec-NC': r'$\overline{\nu}_{e}-NC$', 'tau-CC': r'$\nu_{\tau}-CC$', 'a_tau-CC': r'$\overline{\nu}_{\tau}-CC$'}
    axes.step(bin_edges, hist_1d_energy_accuracy_class_bins_leading_zero, where='pre', linestyle=linestyle, color=color, label=label[particle_type], zorder=3)


def select_class(arr_nn_pred_classes, class_vector, invert=False):
    """
    Selects the rows in an arr_nn_pred_classes array that correspond to a certain class_vector.
    :param arr_nn_pred_classes: array that contains important information for each event (mc_info + model predictions).
    :param (int, int) class_vector: Specifies the class that is used for filtering the array. E.g. (14,1) for muon-CC.
    :param bool invert: Specifies, if the specified class should be selected (standard, False) or deselected (True).
    """
    check_arr_for_class = arr_nn_pred_classes[:, 2:4] == class_vector  # returns a bool for each of the class_vector entries

    # Select only the events, where every bool for one event is True
    indices_rows_with_class = np.logical_and(check_arr_for_class[:, 0], check_arr_for_class[:, 1])
    if invert is True: indices_rows_with_class = np.invert(indices_rows_with_class)
    selected_rows_of_class = arr_nn_pred_classes[indices_rows_with_class]

    return selected_rows_of_class


#-- Functions for making energy to accuracy plots --#

#-- Functions for making probability plots --#

def make_prob_hists(arr_nn_pred, folder_name, modelname, precuts=(False, '3-100_GeV_prod')):
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

    configure_hstack_plot(plot_title='Probability to be classified as shower, 3-40GeV', savepath=folder_name + 'plots/ts_prob_shower_' + modelname)
    plt.cla()

    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'muon-CC', 1, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_muon-CC', 1, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'elec-CC', 1, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_elec-CC', 1, plot_range=(0, 1), color='r', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'elec-NC', 1, plot_range=(0,1), color='saddlebrown', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_elec-NC', 1, plot_range=(0, 1), color='saddlebrown', linestyle='--')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'tau-CC', 1, plot_range=(0,1), color='g', linestyle='-')
    make_prob_hist_class(arr_nn_pred_ecut, axes, particle_types_dict, 'a_tau-CC', 1, plot_range=(0, 1), color='g', linestyle='--')


    configure_hstack_plot(plot_title='Probability to be classified as track, 3-40GeV', savepath=folder_name + 'plots/ts_prob_track_' + modelname)
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
def make_hist_2d_property_vs_property(arr_nn_pred, folder_name, modelname, property_types=('bjorken-y', 'probability'), e_cut=(3, 100), precuts=(False, '3-100_GeV_prod')):
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


    for key in iter(list(particle_types_dict.keys())):
        make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut, key,
                           savepath=folder_name + 'plots/ts_hist_2d_' + key + '_' + property_types[0] + '_vs_'
                                    + property_types[1] + '_e_cut_' + str(e_cut[0]) + '_' + str(e_cut[1]) + '_' + modelname)

    # make multiple cuts in range
    e_cut_range, stepsize = (3, 40), 1

    pdf_pages = {}

    for key in iter(list(particle_types_dict.keys())):
        pdf_pages[key] = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/ts_hist_2d_' + key + '_'
                                                         + property_types[0] + '_vs_' + property_types[1] + '_e_cut_'
                                                           + str(e_cut[0]) + '_' + str(e_cut[1]) + '_multiple_' + modelname + '.pdf')

    for i in range(e_cut_range[1] - e_cut_range[0]):
        e_cut_temp = (e_cut_range[0] + i*stepsize, e_cut_range[0] + i * stepsize + stepsize)
        for key in iter(list(particle_types_dict.keys())):
            make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut_temp, key, pdf_file=pdf_pages[key])

    for i in range(e_cut_range[1] - e_cut_range[0]):
        e_cut_temp = (e_cut_range[0] + i*stepsize, e_cut_range[0] + i * stepsize + stepsize)
        for key in iter(list(particle_types_dict.keys())):
            make_hist_2d_class(prop_1, prop_2, arr_nn_pred, particle_types_dict, e_cut_temp, key, pdf_file=pdf_pages[key], log=True)

    for key in iter(list(pdf_pages.keys())):
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


def calculate_and_plot_separation_pid(arr_nn_pred, folder_name, modelname, precuts=(False, '3-100_GeV_prod')):
    """
    Calculates and plots the separability (1-c) plot.
    :param ndarray(ndim=2) arr_nn_pred: array that contains important information for each event (mc_info + model predictions).
    :param str modelname: Name of the model used for plot savenames.
    :param tuple precuts: Boolean flag that specifies if only events that survive the Pheid precuts should be used in making the plots. # TODO fix docs
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, precuts=precuts[1], invert=False)

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1),
                           'elec-NC': (12, 0), 'a_elec-NC': (-12, 0)}

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
        for j in range(bins - 1):
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

    plt.savefig(folder_name + 'plots/ts_Correlation_Coefficients_' + modelname + '.pdf')
    plt.savefig(folder_name + 'plots/ts_Correlation_Coefficients_' + modelname + '.png', dpi=600)

    plt.close()


#- Classification -#

#- Regression -#

def make_2d_energy_resolution_plot(arr_nn_pred, modelname, folder_name, energy_bins=np.arange(1,101,1), precuts=(False, '3-100_GeV_prod'), correct_energy=(False, 'median')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param energy_bins:
    :param precuts:
    :param correct_energy:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])
    if correct_energy[0] is True:
        arr_nn_pred = correct_reco_energy(arr_nn_pred, metric=correct_energy[1])

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    energy_mc = arr_nn_pred[:, 4] # TODO make config file to get the indices
    energy_pred = arr_nn_pred[:, 9]

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/energy_resolution_' + modelname + '.pdf')

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        hist_2d_energy_ic = np.histogram2d(energy_mc[is_ic], energy_pred[is_ic], energy_bins)
        bin_edges_energy = hist_2d_energy_ic[1]

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        energy_res_ic = ax.pcolormesh(bin_edges_energy, bin_edges_energy, hist_2d_energy_ic[0].T,
                                     norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_energy_ic[0].T.max()))

        reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
        title = plt.title(reco_name + ic_list[ic]['title'])
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(energy_res_ic, ax=ax)
        cbar.ax.set_ylabel('Number of events')
        y_label = 'Corrected reconstructed energy (GeV)' if correct_energy[0] is True and ic not in ['muon-CC'] else 'Reconstructed energy (GeV)'
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel(y_label)
        plt.tight_layout()

        pdf_plots.savefig(fig)
        cbar.remove() # TODO still necessary?
        ax.cla()

    pdf_plots.close()
    plt.close()


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
    # TODO fix taus
    abs_particle_type = np.abs(particle_type)
    track = np.logical_and(abs_particle_type == 14, is_cc == True)
    shower = np.logical_or(abs_particle_type == 16, abs_particle_type == 12)

    return track, shower


def get_boolean_interaction_channel_separation(particle_type, is_cc, interaction_channel):
    """

    :param particle_type:
    :param is_cc:
    :param str interaction_channel:
    :return:
    """
    ic_dict = {'muon-CC': (14, 1), 'elec-CC': (12, 1), 'elec-NC': (12, 0), 'tau-CC': (16, 1)}
    if interaction_channel not in list(ic_dict.keys()):
        raise ValueError('The interaction_channel ' + str(interaction_channel) + ' is not known.')

    abs_ptype = np.abs(particle_type)
    boolean_interaction_channel = np.logical_and(abs_ptype == ic_dict[interaction_channel][0], is_cc == ic_dict[interaction_channel][1])

    return boolean_interaction_channel


def correct_reco_energy(arr_nn_pred, metric='median'):
    """
    Evaluates the correction factors based on e-CC events, applies them to ALL shower events (e-CC, e-NC, tau-CC)
    :param arr_nn_pred:
    :param metric:
    :return:
    """
    l = get_array_indices()
    # label_indices = {'run_id': 0, 'event_id': 1, 'particle_type': 2, 'is_cc': 3, 'bjorken_y': {'true_index': 5, 'reco_index': 13},
    #                  'energy': {'true_index': 4, 'reco_index': 9, 'index_label_std': 15, 'str': ('energy', 'GeV')},
    #                  'dir_x': {'true_index':6, 'reco_index': 10, 'reco_index_std': 17, 'str': ('dir_x', 'rad')},
    #                  'dir_y': {'true_index':7, 'reco_index': 11, 'reco_index_std': 19, 'str': ('dir_y', 'rad')},
    #                  'dir_z': {'true_index':8, 'reco_index': 12, 'reco_index_std': 21, 'str': ('dir_z', 'rad')},
    #                  'vtx_x': {'true_index': 9},
    #                  'vtx_y': {'true_index': 10},
    #                  'vtx_z': {'true_index': 11},
    #                  'time_residual_vtx': {'true_index': 0}}

    is_track, is_shower = get_boolean_track_and_shower_separation(arr_nn_pred[:, l['particle_type']], arr_nn_pred[:, l['is_cc']])
    is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, l['particle_type']], arr_nn_pred[:, l['is_cc']], 'elec-CC')

    energy_mc, energy_pred = arr_nn_pred[:, l['energy']['true_index']][is_ic], arr_nn_pred[:, l['energy']['reco_index']][is_ic]

    arr_nn_pred_corr = np.copy(arr_nn_pred) # TODO still necessary?

    correction_factors_x, correction_factors_y = [], []

    e_range = np.logspace(np.log(3)/np.log(2),np.log(100)/np.log(2),50,base=2)
    n_ranges = e_range.shape[0] - 1
    for i in range(n_ranges):
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
    #correction_factor_en_pred = np.interp(energy_pred, correction_factors_x, correction_factors_y)
    energy_pred_orig_shower = arr_nn_pred[:, l['energy']['reco_index']][is_shower]
    correction_factor_en_pred = np.interp(energy_pred_orig_shower, correction_factors_x, correction_factors_y)

    # apply correction to ALL shower ic's (including all taus atm)
    arr_nn_pred_corr[:, l['energy']['reco_index']][is_shower] = energy_pred_orig_shower + (- correction_factor_en_pred) * energy_pred_orig_shower

    return arr_nn_pred_corr


def make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median_relative', energy_bins=np.linspace(1,100,32),
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

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/energy_resolution_' + modelname + '.pdf')

    for ic in ic_list.keys():

        energy_metric_plot_data_ic = calc_plot_data_of_energy_dependent_label(arr_nn_pred, ic, energy_bins=energy_bins, label=('energy', metric))
        if energy_metric_plot_data_ic is None: continue # ic not contained in the arr_nn_pred

        bins, metric_ic = energy_metric_plot_data_ic[0], energy_metric_plot_data_ic[1]

        ax.step(bins, metric_ic, linestyle="-", where='post', label='OrcaNet')

        if compare_shallow[0] is True:
            energy_metric_plot_data_shallow_ic = calc_plot_data_of_energy_dependent_label(arr_nn_pred_shallow, ic, energy_bins=energy_bins,
                                                                                          label=('energy', metric))
            ax.step(bins, energy_metric_plot_data_shallow_ic[1], linestyle="-", where='post', label='Standard Reco')

        reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
        x_ticks_major = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks_major)
        ax.minorticks_on()
        title = plt.title(reco_name + ic_list[ic]['title'])
        title.set_position([.5, 1.04])
        y_label = 'Median relative error (corrected energy)' if correct_energy[0] is True and ic not in ['muon-CC'] else 'Median relative error (energy)'
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel(y_label)

        ax.grid(True)
        ax.legend(loc='upper right')

        pdf_plots.savefig(fig)
        ax.cla()

    plt.close()
    pdf_plots.close()


def calc_plot_data_of_energy_dependent_label(arr_nn_pred, interaction_channel, energy_bins=np.linspace(1, 100, 20), label=('energy', 'median_relative')):
    """
    Generate binned statistics for the energy mae, or the relative mae. Separately for different interaction channels.
    :param arr_nn_pred:
    :param interaction_channel
    :param energy_bins:
    :param label:
    :return:
    """
    is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], interaction_channel)
    if bool(np.any(is_ic, axis=0)) is False:
        return None # if ic is not contained in the arr_nn_pred
    else:
        arr_nn_pred = arr_nn_pred[is_ic]

    labels = {'energy': {'reco_index': 9, 'mc_index': 4}, 'dir_x': {'reco_index': 10, 'mc_index': 6},
              'dir_y': {'reco_index': 11, 'mc_index': 7}, 'dir_z': {'reco_index': 12, 'mc_index': 8},
              'azimuth': {'reco_index': None, 'mc_index': None}, 'zenith': {'reco_index': None, 'mc_index': None},
              'bjorken_y': {'reco_index': 13, 'mc_index': 5}}
    label_name, metric = label[0], label[1]

    if label_name == 'azimuth' or label_name == 'zenith':
        dir_mc_index_range = (labels['dir_x']['mc_index'], labels['dir_z']['mc_index'] + 1)
        dir_pred_index_range = (labels['dir_x']['reco_index'],  labels['dir_z']['reco_index'] + 1)
        dir_mc = arr_nn_pred[:, dir_mc_index_range[0] : dir_mc_index_range[1]]
        dir_pred = arr_nn_pred[:, dir_pred_index_range[0]: dir_pred_index_range[1]]

        if label_name == 'azimuth':
            # atan2(y,x)
            mc_label, reco_label = np.arctan2(dir_mc[:, 1], dir_mc[:, 0]), np.arctan2(dir_pred[:, 1], dir_pred[:, 0])  # atan2(y,x)
        else: # zenith
            # atan2(z, sqrt(x**2 + y**2))
            mc_label = np.arctan2(dir_mc[:, 2], np.sqrt(np.power(dir_mc[:, 0], 2) + np.power(dir_mc[:, 1], 2)))
            reco_label = np.arctan2(dir_pred[:, 2], np.sqrt(np.power(dir_pred[:, 0], 2) + np.power(dir_pred[:, 1], 2)))

    else:
        mc_label = arr_nn_pred[:, labels[label_name]['mc_index']]
        reco_label = arr_nn_pred[:, labels[label_name]['reco_index']] # reconstruction results for the chosen label

    if label_name  in ['energy', 'bjorken_y', 'azimuth', 'zenith']:
        err = np.abs(reco_label - mc_label)

    elif label_name in ['dir_x', 'dir_y', 'dir_z']:
        reco_label_list = []
        for i in range(reco_label.shape[0]): # TODO remove, not necessary?
            dir = reco_label[i]
            if dir < -1: dir = -1
            if dir > 1: dir = 1
            reco_label_list.append(dir)

        reco_label = np.array(reco_label_list)
        err = np.abs(reco_label - mc_label)

    else:
        raise ValueError('The label' + str(label[0]) + ' is not available.')

    mc_energy = arr_nn_pred[:, labels['energy']['mc_index']]
    energy_to_label_performance_plot_data_ic = bin_error_in_energy_bins(energy_bins, mc_energy, err, operation=metric)

    return energy_to_label_performance_plot_data_ic


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
    for bin_no in range(1, len(energy_bins)):
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


def make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, folder_name, energy_bins=np.linspace(1,100,49), precuts=(False, '3-100_GeV_prod'),
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

    if compare_shallow[0] is True:
        arr_nn_pred_shallow = compare_shallow[1]
        if precuts[0] is True:
            arr_nn_pred_shallow = arr_nn_pred_select_pheid_events(arr_nn_pred_shallow, invert=False, precuts=precuts[1])

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/energy_std_rel_' + modelname + '.pdf')

    for ic in ic_list.keys():
        std_rel_ic = get_std_rel_plot_data(arr_nn_pred, ic, energy_bins)
        if std_rel_ic is None: continue  # ic not contained in the arr_nn_pred

        ax.step(energy_bins, std_rel_ic, linestyle="-", where='post', label='OrcaNet')

        if compare_shallow[0] is True:
            std_rel_ic_shallow= get_std_rel_plot_data(arr_nn_pred_shallow, ic, energy_bins)
            ax.step(energy_bins, std_rel_ic_shallow, linestyle="-", where='post', label='Standard Reco')

        reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
        x_ticks_major = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks_major)
        ax.minorticks_on()
        title = plt.title(reco_name + ic_list[ic]['title'])
        title.set_position([.5, 1.04])
        y_label = r'$\sigma / E_{true}$ (corrected energy)' if correct_energy[0] is True and ic not in ['muon-CC'] else r'$\sigma / E_{true}$'
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel(y_label)
        ax.grid(True)

        ax.legend(loc='upper right')
        pdf_plots.savefig(fig)
        ax.cla()

    plt.close()
    pdf_plots.close()


def get_std_rel_plot_data(arr_nn_pred, ic, energy_bins):
    """

    :param arr_nn_pred:
    :param ic
    :param energy_bins:
    :return:
    """
    energy_mc = arr_nn_pred[:, 4]
    energy_pred = arr_nn_pred[:, 9]

    is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
    if bool(np.any(is_ic, axis=0)) is False:
        return None # if ic is not contained in the arr_nn_pred

    energy_mc_ic, energy_pred_ic = energy_mc[is_ic], energy_pred[is_ic]

    std_rel_ic = [] # y-axis of the plot
    for i in range(energy_bins.shape[0] -1):
        e_range_low, e_range_high = energy_bins[i], energy_bins[i+1]
        e_range_mean = (e_range_low + e_range_high)/ float(2)

        e_pred_ic_cut_boolean = np.logical_and(e_range_low < energy_mc_ic, energy_mc_ic <= e_range_high)
        e_pred_ic_cut = energy_pred_ic[e_pred_ic_cut_boolean]

        std_ic_temp = np.std(e_pred_ic_cut)
        std_rel_ic.append(std_ic_temp / float(e_range_mean))

    # fix for mpl
    std_rel_ic.append(std_rel_ic[-1])

    return std_rel_ic


def make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', energy_bins=np.linspace(1,100,32),
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

    #directions = ['dir_x', 'dir_y', 'dir_z', 'azimuth', 'zenith']
    directions = {'vector': ['dir_x', 'dir_y', 'dir_z'], 'spherical': ['azimuth', 'zenith']}
    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/dir_resolution_' + modelname + '.pdf')
    reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '

    for ic in ic_list.keys():
        for key, list_dirs in directions.items():
            dir_plot_data_ic = {}

            for dir_coord in list_dirs:
                dir_plot_data_ic[dir_coord] = calc_plot_data_of_energy_dependent_label(arr_nn_pred, ic, energy_bins=energy_bins,
                                                                            label=(dir_coord, metric))
            # continue if the specified interaction channel is not contained in any element of the dict
            if dir_plot_data_ic[list(dir_plot_data_ic.keys())[0]] is None: continue

            for dir_coord in list_dirs:
                bins, dir_perf_ic = dir_plot_data_ic[dir_coord][0], dir_plot_data_ic[dir_coord][1]
                ax.step(bins, dir_perf_ic, linestyle="-", where='post', label='DL ' + dir_coord)

                if compare_shallow[0] is True:
                    dir_plot_data_shallow_ic = calc_plot_data_of_energy_dependent_label(arr_nn_pred_shallow, ic, energy_bins=energy_bins,
                                                                                        label=(dir_coord, metric))
                    ax.step(bins, dir_plot_data_shallow_ic[1], linestyle="-", where='post', label='Std ' + dir_coord)

            x_ticks_major = np.arange(0, 101, 10)
            ax.set_xticks(x_ticks_major)
            ax.minorticks_on()
            title = plt.title(reco_name + ic_list[ic]['title'])
            title.set_position([.5, 1.04])
            ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median error dir')
            ax.grid(True)
            ax.legend(loc='upper right')

            pdf_plots.savefig(fig)
            ax.cla()

    plt.close()
    pdf_plots.close()


def make_2d_dir_correlation_plot(arr_nn_pred, modelname, folder_name, precuts=(False, '3-100_GeV_prod')):
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
    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/dir_correlation_' + modelname + '.pdf')

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        ic_title = ic_list[ic]['title']

        # dir_x, dir_y, dir_z plots
        dir_bins = np.linspace(-1, 1, 100)
        plot_2d_dir_correlation(arr_nn_pred, is_ic, ic_title, 'dir_x', labels, dir_bins, fig, ax, pdf_plots, modelname)
        plot_2d_dir_correlation(arr_nn_pred, is_ic, ic_title, 'dir_y', labels, dir_bins, fig, ax, pdf_plots, modelname)
        plot_2d_dir_correlation(arr_nn_pred, is_ic, ic_title, 'dir_z', labels, dir_bins, fig, ax, pdf_plots, modelname)

        # azimuth and zenith plots
        pi = math.pi
        dir_bins_azimuth, dir_bins_zenith = np.linspace(-pi, pi, 100), np.linspace(-pi/float(2), pi/float(2), 100)
        plot_2d_dir_correlation(arr_nn_pred, is_ic, ic_title, 'azimuth', labels, dir_bins_azimuth, fig, ax, pdf_plots, modelname)
        plot_2d_dir_correlation(arr_nn_pred, is_ic, ic_title, 'zenith', labels, dir_bins_zenith, fig, ax, pdf_plots, modelname)

    plt.close()
    pdf_plots.close()


def plot_2d_dir_correlation(arr_nn_pred, is_ic, ic_title, label, labels, dir_bins, fig, ax, pdf_plots, modelname):
    """

    :param arr_nn_pred:
    :param is_ic:
    :param ic_title:
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

    if label == 'azimuth' or label == 'zenith':
        dir_mc_index_range = (labels['dir_x']['mc_index'], labels['dir_z']['mc_index'] + 1)
        dir_pred_index_range = (labels['dir_x']['reco_index'],  labels['dir_z']['reco_index'] + 1)
        dir_mc = arr_nn_pred[:, dir_mc_index_range[0] : dir_mc_index_range[1]]
        dir_pred = arr_nn_pred[:, dir_pred_index_range[0]: dir_pred_index_range[1]]

    else:
        dir_mc = arr_nn_pred[:, labels[label]['mc_index']]
        dir_pred = arr_nn_pred[:, labels[label]['reco_index']]

    if label == 'azimuth':
        dir_mc, dir_pred = np.arctan2(dir_mc[:, 1], dir_mc[:, 0]), np.arctan2(dir_pred[:, 1], dir_pred[:, 0]) # atan2(y,x)
        plt.plot([-math.pi,math.pi], [-math.pi,math.pi], 'k-', lw=1, zorder=10)

    if label == 'zenith':
        # atan2(z, sqrt(x**2 + y**2))
        dir_mc = np.arctan2(dir_mc[:, 2], np.sqrt(np.power(dir_mc[:, 0], 2) + np.power(dir_mc[:, 1], 2)))
        dir_pred = np.arctan2(dir_pred[:, 2], np.sqrt(np.power(dir_pred[:, 0], 2) + np.power(dir_pred[:, 1], 2)))

    hist_2d_dir_ic = np.histogram2d(dir_mc[is_ic], dir_pred[is_ic], dir_bins)
    bin_edges_dir = hist_2d_dir_ic[1]

    dir_corr_ic = ax.pcolormesh(bin_edges_dir, bin_edges_dir, hist_2d_dir_ic[0].T,
                                     norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_dir_ic[0].T.max()))

    plot_line_through_the_origin(label)

    title = plt.title(reco_name + ic_title)
    title.set_position([.5, 1.04])
    cbar = fig.colorbar(dir_corr_ic, ax=ax)
    cbar.ax.set_ylabel('Number of events')
    ax.set_xlabel('True direction [' + label + ']'), ax.set_ylabel('Reconstructed direction [' + label + ']')
    plt.tight_layout()

    pdf_plots.savefig(fig)
    cbar.remove() # TODO still necessary or cleaned by plt.cla()?

    plt.cla()


def plot_line_through_the_origin(label): # TODO add to all 2d plots
    """

    :param label:
    :return:
    """
    pi = math.pi

    if label == 'azimuth':
        plt.plot([-pi, pi], [-pi, pi], 'k-', lw=1, zorder=10)

    elif label == 'zenith':
        plt.plot([-pi/float(2),pi/float(2)], [-pi/float(2),pi/float(2)], 'k-', lw=1, zorder=10)

    elif label == 'bjorken-y':
        plt.plot([0,1], [0,1], 'k-', lw=1, zorder=10)

    else:
        plt.plot([-1,1], [-1,1], 'k-', lw=1, zorder=10)


def make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', energy_bins=np.linspace(1,100,32),
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
        # correct by to 1 for e-NC events
        abs_particle_type, is_cc = np.abs(arr_nn_pred_shallow[:, 2]), arr_nn_pred_shallow[:, 3]
        is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)
        arr_nn_pred_shallow[is_e_nc, 5] = 1

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Shower like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/bjorken_y_' + modelname + '.pdf')

    # correct by to 1 for e-NC events
    abs_particle_type, is_cc = np.abs(arr_nn_pred[:, 2]), arr_nn_pred[:, 3]
    is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)
    arr_nn_pred[is_e_nc, 5] = 1

    for ic in ic_list.keys():

        by_metric_plot_data_ic = calc_plot_data_of_energy_dependent_label(arr_nn_pred, ic, energy_bins=energy_bins, label=('bjorken_y', metric))
        if by_metric_plot_data_ic is None: continue

        bins, metric_ic = by_metric_plot_data_ic[0], by_metric_plot_data_ic[1]
        ax.step(bins, metric_ic, linestyle="-", where='post', label='OrcaNet')

        if compare_shallow[0] is True:
            by_metric_plot_data_shallow_ic = calc_plot_data_of_energy_dependent_label(arr_nn_pred_shallow, ic, energy_bins=energy_bins,
                                                                                      label=('bjorken_y', metric))
            print(by_metric_plot_data_shallow_ic[1]) # TODO check what happens if e-NC
            ax.step(bins, by_metric_plot_data_shallow_ic[1], linestyle="-", where='post', label='Standard Reco')

        reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
        x_ticks_major = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks_major)
        ax.minorticks_on()
        title = plt.title(reco_name + ic_list[ic]['title'])
        title.set_position([.5, 1.04])
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel('Median error bjorken-y')
        ax.grid(True)
        ax.legend(loc='upper right')

        pdf_plots.savefig(fig)
        ax.cla()

    plt.close()
    pdf_plots.close()


def make_2d_bjorken_y_resolution_plot(arr_nn_pred, modelname, folder_name, by_bins=np.linspace(0,1,101), precuts=(False, '3-100_GeV_prod')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param by_bins:
    :param precuts:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/bjorken-y_resolution_' + modelname + '.pdf')

    #arr_nn_pred = arr_nn_pred[arr_nn_pred[:, 4] > 3]

    # correct by to 1 for e-NC events
    abs_particle_type, is_cc = np.abs(arr_nn_pred[:, 2]), arr_nn_pred[:, 3]
    is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)
    arr_nn_pred[is_e_nc, 5] = 1

    by_mc = arr_nn_pred[:, 5]
    by_pred = arr_nn_pred[:, 13]

    for ic in ic_list.keys():

        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        hist_2d_by_ic = np.histogram2d(by_mc[is_ic], by_pred[is_ic], by_bins)
        bin_edges_by = hist_2d_by_ic[1]

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        by_res_ic = ax.pcolormesh(bin_edges_by, bin_edges_by, hist_2d_by_ic[0].T,
                                         norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_by_ic[0].T.max()))

        plot_line_through_the_origin('bjorken-y')

        reco_name = 'OrcaNet: ' if modelname != 'shallow_reco' else 'Standard Reco: '
        title = plt.title(reco_name + ic_list[ic]['title'])
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(by_res_ic, ax=ax)
        cbar.ax.set_ylabel('Number of events')
        ax.set_xlabel('True bjorken-y'), ax.set_ylabel('Reconstructed bjorken-y (GeV)')
        plt.tight_layout()

        pdf_plots.savefig(fig)
        cbar.remove() # TODO still necessary or cleaned by plt.cla()?

        ax.cla()

    plt.close()
    pdf_plots.close()


# error plots

def make_1d_reco_err_div_by_std_plot(arr_nn_pred, modelname, folder_name, precuts=(False, '3-100_GeV_prod')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param precuts:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])

    # do this for energy, bj-y, dir_x, dir_y, dir_z
    # 1d (y_true - y_reco) / std_reco
    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/1d_std_errors_reco_err_div_by_std_' + modelname + '.pdf')

    plot_1d_reco_err_div_by_std_for_label(arr_nn_pred, fig, ax, pdf_plots, 'energy')
    plot_1d_reco_err_div_by_std_for_label(arr_nn_pred, fig, ax, pdf_plots, 'bjorken-y')
    plot_1d_reco_err_div_by_std_for_label(arr_nn_pred, fig, ax, pdf_plots, 'dir_x')
    plot_1d_reco_err_div_by_std_for_label(arr_nn_pred, fig, ax, pdf_plots, 'dir_y')
    plot_1d_reco_err_div_by_std_for_label(arr_nn_pred, fig, ax, pdf_plots, 'dir_z')

    plt.close()
    pdf_plots.close()


def plot_1d_reco_err_div_by_std_for_label(arr_nn_pred, fig, ax, pdf_plots, label):
    """

    :param arr_nn_pred:
    :param fig:
    :param ax:
    :param pdf_plots:
    :param label:
    :return:
    """
    labels = {'energy': {'index_label_true': 4, 'index_label_pred': 9, 'index_label_std': 15},
              'bjorken-y': {'index_label_true': 5, 'index_label_pred': 13, 'index_label_std': 23},
              'dir_x': {'index_label_true': 6, 'index_label_pred': 10, 'index_label_std': 17},
              'dir_y': {'index_label_true': 7, 'index_label_pred': 11, 'index_label_std': 19},
              'dir_z': {'index_label_true': 8, 'index_label_pred': 12, 'index_label_std': 21}}

    label_true = arr_nn_pred[:, labels[label]['index_label_true']]
    label_pred = arr_nn_pred[:, labels[label]['index_label_pred']]
    label_std_pred = arr_nn_pred[:, labels[label]['index_label_std']]

    label_std_pred = label_std_pred * 1.253 # TODO correction with mse training

    pred_err_div_by_std = np.divide(label_true - label_pred, np.abs(label_std_pred))
    exclude_outliers = np.logical_and(pred_err_div_by_std > -10, pred_err_div_by_std < 10)

    print(np.std(pred_err_div_by_std[exclude_outliers]))

    #plt.hist(pred_err_div_by_std, bins=100, label=label, range=(-0.5, 0.5))
    plt.hist(pred_err_div_by_std[exclude_outliers], bins=100, label=label)

    title = plt.title('Gausian Likelihood errors for ' + label)
    title.set_position([.5, 1.04])
    ax.set_xlabel(r'$(y_{\mathrm{true}} - y_{\mathrm{pred}})/ \sigma_{\mathrm{pred}}$'), ax.set_ylabel('Counts [#]')
    ax.grid(True)

    pdf_plots.savefig(fig)
    ax.cla()


def make_1d_reco_err_to_reco_residual_plot(arr_nn_pred, modelname, folder_name, precuts=(False, '3-100_GeV_prod')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param precuts:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/1d_reco_errors_to_reco_residual_' + modelname + '.pdf')

    # correct by to 1 for e-NC events
    abs_particle_type, is_cc = np.abs(arr_nn_pred[:, 2]), arr_nn_pred[:, 3]
    is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)
    arr_nn_pred[is_e_nc, 5] = 1

    n_x_bins = 50
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'energy')
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'bjorken-y')
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'dir_x')
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'dir_y')
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'dir_z')
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'azimuth')
    plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, 'zenith')

    plt.close()
    pdf_plots.close()


def plot_1d_reco_err_to_reco_residual_for_label(arr_nn_pred, n_x_bins, fig, ax, pdf_plots, label):
    """

    :param arr_nn_pred:
    :param n_x_bins:
    :param fig:
    :param ax:
    :param pdf_plots:
    :param label:
    :return:
    """
    labels = {'energy': {'index_label_true': 4, 'index_label_pred': 9, 'index_label_std': 15},
              'bjorken-y': {'index_label_true': 5, 'index_label_pred': 13, 'index_label_std': 23},
              'dir_x': {'index_label_true': 6, 'index_label_pred': 10, 'index_label_std': 17},
              'dir_y': {'index_label_true': 7, 'index_label_pred': 11, 'index_label_std': 19},
              'dir_z': {'index_label_true': 8, 'index_label_pred': 12, 'index_label_std': 21}}

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        if label == 'azimuth' or label == 'zenith':
            dx_true, dx_pred = arr_nn_pred[is_ic][:, labels['dir_x']['index_label_true']], arr_nn_pred[is_ic][:, labels['dir_x']['index_label_pred']]
            dy_true, dy_pred = arr_nn_pred[is_ic][:, labels['dir_y']['index_label_true']], arr_nn_pred[is_ic][:,labels['dir_y']['index_label_pred']]
            dz_true, dz_pred = arr_nn_pred[is_ic][:, labels['dir_z']['index_label_true']], arr_nn_pred[is_ic][:, labels['dir_z']['index_label_pred']]

            dx_std_pred = arr_nn_pred[is_ic][:, labels['dir_x']['index_label_std']]
            dy_std_pred = arr_nn_pred[is_ic][:, labels['dir_y']['index_label_std']]
            dz_std_pred = arr_nn_pred[is_ic][:, labels['dir_z']['index_label_std']]

            correction = 1.253
            if label == 'azimuth':
                # atan2(y,x)
                label_true = np.arctan2(dy_true, dx_true)
                label_pred = np.arctan2(dy_pred, dx_pred)
                # print np.amin(label_true), np.amax(label_true)
                # print np.amin(label_pred), np.amax(label_pred)
                # print np.amin(label_true - label_pred), np.amax(label_true - label_pred)

                # TODO does this even make sense??
                #label_std_pred = np.arctan2(np.abs(dy_std_pred * correction), np.abs(dx_std_pred * correction))
                # print np.amin(dx_std_pred), np.amax(dx_std_pred)
                dx_pred, dy_pred = np.clip(dx_pred, -0.999, 0.999), np.clip(dy_pred, -0.999, 0.999)
                dx_std_pred, dy_std_pred = np.clip(dx_std_pred, 0, None) * correction, np.clip(dy_std_pred, 0, None) * correction

                # error propagation of atan2 with correlations between y and x neglected (covariance term = 0)
                label_std_pred = np.sqrt(( dx_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dy_std_pred ** 2 +
                                         (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dx_std_pred ** 2)
                n = label_std_pred > math.pi
                percentage_true = np.sum(n)/float(label_std_pred.shape[0]) * 100
                print(np.sum(n))
                print(percentage_true)
                print(dz_pred[n])
                print(dz_true[n])

                label_std_pred = np.clip(label_std_pred, None, math.pi)

                # we probably can't neglect the covariance term in the error propagation, since y and x are correlated
                # tested: doesn't really make a difference
                # mean_dx, mean_dy = np.mean(dx_pred), np.mean(dy_pred)
                # covariance_xy = 1/float(dx_pred.shape[0]) * np.sum((dx_pred - mean_dx) * (dy_pred - mean_dy))
                # test = np.abs(2 * (dx_pred / (dx_pred ** 2 + dy_pred ** 2)) * (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) * covariance_xy)
                # label_std_pred = np.sqrt(np.abs(( dx_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dy_std_pred ** 2 +
                #                          (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dx_std_pred ** 2 +
                #                          2 * (dx_pred / (dx_pred ** 2 + dy_pred ** 2)) * (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) * covariance_xy))
                # label_std_pred = np.clip(label_std_pred, None, math.pi/float(2))

                # Steffen's formula
                # zenith_pred = np.arctan2(dz_pred, np.sqrt(np.power(dx_pred, 2) + np.power(dy_pred, 2)))
                # label_std_pred = np.sqrt(dx_std_pred ** 2 + dy_std_pred ** 2) / np.sin(zenith_pred + math.pi/float(2))
                #label_std_pred = np.clip(label_std_pred, None, math.pi)

                print('----- AZIMUTH -----')
                #print np.amin(zenith_pred), np.amax(zenith_pred)
                print(np.amin(label_std_pred), np.amax(label_std_pred))
                print(np.mean(label_std_pred))
                print('----- AZIMUTH -----')

            else: # zenith
                # atan2(z, sqrt(x**2 + y**2))
                label_true = np.arctan2(dz_true, np.sqrt(np.power(dx_true, 2) + np.power(dy_true, 2)))
                label_pred = np.arctan2(dz_pred, np.sqrt(np.power(dx_pred, 2) + np.power(dy_pred, 2)))

                # TODO does this even make sense?? Answer: no
                #dx_std_pred, dy_std_pred, dz_std_pred = np.abs(dx_std_pred * correction), np.abs(dy_std_pred * correction), np.abs(dz_std_pred * correction)
                dz_std_pred = np.clip(dz_std_pred, 0, None) * correction
                dz_pred = np.clip(dz_pred, -0.999, 0.999)

                #label_std_pred = np.arctan2(dz_std_pred, np.sqrt(np.power(dx_std_pred, 2) + np.power(dy_std_pred, 2)))
                label_std_pred = np.sqrt((-1 / np.sqrt((1 - dz_pred ** 2))) ** 2 * dz_std_pred ** 2) # zen = arccos(z/norm(r))

                # dx_std_pred, dy_std_pred = np.clip(dx_std_pred, 0, None) * correction, np.clip(dy_std_pred, 0, None) * correction
                # dx_pred, dy_pred = np.clip(dx_pred, -0.999, 0.999), np.clip(dy_pred, -0.999, 0.999)
                # label_std_pred = np.sqrt( (np.sqrt(dx_pred ** 2 + dy_pred ** 2) / (dx_pred ** 2 + dy_pred ** 2 + dz_pred ** 2))**2 * dz_std_pred ** 2 +
                #                           ((- dx_pred * dz_pred) / (np.sqrt(dx_pred ** 2 + dy_pred ** 2) * (dx_pred ** 2 + dy_pred ** 2 + dz_pred ** 2))) ** 2 * dx_std_pred ** 2 +
                #                           ((- dy_pred * dz_pred) / (np.sqrt(dx_pred ** 2 + dy_pred ** 2) * (dx_pred ** 2 + dy_pred ** 2 + dz_pred ** 2))) ** 2 * dy_std_pred ** 2)

                print('----- ZENITH -----')
                print(np.amin(label_true - label_pred), np.amax(label_true - label_pred))
                print(np.amin(label_std_pred), np.amax(label_std_pred))

                print(label_std_pred)
                print('----- ZENITH -----')

        else:
            label_true = arr_nn_pred[is_ic][:, labels[label]['index_label_true']]
            label_pred = arr_nn_pred[is_ic][:, labels[label]['index_label_pred']]
            label_std_pred = arr_nn_pred[is_ic][:, labels[label]['index_label_std']]

            label_std_pred = label_std_pred * 1.253  # TODO correction with mse training
            label_std_pred = np.abs(label_std_pred) #TODO necessary? -> rather set to zero, regarding the loss function?

        std_pred_range = (np.amin(label_std_pred), np.amax(label_std_pred))
        x_bins_std_pred = np.linspace(std_pred_range[0], std_pred_range[1], n_x_bins + 1)

        x, y = [], []
        for i in range(x_bins_std_pred.shape[0] - 1): # same as n_x_bins
            label_std_pred_low, label_std_pred_high = x_bins_std_pred[i], x_bins_std_pred[i+1]
            label_std_pred_mean = (label_std_pred_low + label_std_pred_high) / float(2)

            label_std_pred_cut_boolean = np.logical_and(label_std_pred_low < label_std_pred, label_std_pred <= label_std_pred_high)
            if np.count_nonzero(label_std_pred_cut_boolean) < 100:
                continue

            if label == 'azimuth':
                # if abs(residual_azimuth) <= pi, take az_true - az_pred ; if residual_azimuth > pi take
                all_residuals = label_true[label_std_pred_cut_boolean] - label_pred[label_std_pred_cut_boolean]
                larger_pi = np.abs(all_residuals) > math.pi
                sign_larger_pi = np.sign(all_residuals[larger_pi]) # loophole for residual == 0, but doesn't matter below
                all_residuals[larger_pi] = sign_larger_pi * 2 * math.pi - (all_residuals[larger_pi]) # need same sign for 2pi compared to all_residuals value
                #print np.amin(all_residuals), np.amax(all_residuals)

                residuals_std = np.std(all_residuals)

            else:
                residuals_std = np.std(label_true[label_std_pred_cut_boolean] - label_pred[label_std_pred_cut_boolean])

            x.append(label_std_pred_mean)
            y.append(residuals_std)

        plt.scatter(x, y, s=20, lw=0.75, c='blue', marker='+')

        title = plt.title(ic_list[ic]['title'] + ': ' + label)
        title.set_position([.5, 1.04])
        ax.set_xlabel(r'Estimated uncertainty $\sigma_{pred}$ [GeV]'), ax.set_ylabel('Standard deviation of residuals [GeV]') # TODO fix units
        ax.grid(True)

        pdf_plots.savefig(fig)
        ax.cla()


def make_2d_dir_correlation_plot_different_sigmas(arr_nn_pred, modelname, folder_name, precuts=(False, '3-100_GeV_prod')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param dir_bins:
    :param precuts:
    :return:
    """
    if precuts[0] is True:
        arr_nn_pred = arr_nn_pred_select_pheid_events(arr_nn_pred, invert=False, precuts=precuts[1])

    labels = {'energy': {'reco_index': 9, 'mc_index': 4, 'index_label_std': 15, 'str': ('energy', 'GeV')},
              'dir_x': {'reco_index': 10, 'mc_index':6, 'reco_index_std': 17, 'str': ('dir_x', 'rad')},
              'dir_y': {'reco_index': 11, 'mc_index':7, 'reco_index_std': 19, 'str': ('dir_y', 'rad')},
              'dir_z': {'reco_index': 12, 'mc_index':8, 'reco_index_std': 21, 'str': ('dir_z', 'rad')}}

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(folder_name + 'plots/correlation_diff_sigmas_' + modelname + '.pdf')

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        ic_title = ic_list[ic]['title']

        # azimuth and zenith plots
        pi = math.pi
        dir_bins_azimuth, dir_bins_zenith = np.linspace(-pi, pi, 100), np.linspace(-pi/float(2), pi/float(2), 100)
        plot_2d_dir_correlation_different_sigmas(arr_nn_pred, is_ic, ic_title, 'azimuth', labels, dir_bins_azimuth, fig, ax, pdf_plots)
        plot_2d_dir_correlation_different_sigmas(arr_nn_pred, is_ic, ic_title, 'zenith', labels, dir_bins_zenith, fig, ax, pdf_plots)

        energy_bins = np.linspace(1, 100, 100)
        plot_2d_dir_correlation_different_sigmas(arr_nn_pred, is_ic, ic_title, 'energy', labels, energy_bins, fig, ax, pdf_plots)

    plt.close()
    pdf_plots.close()


def plot_2d_dir_correlation_different_sigmas(arr_nn_pred, is_ic, ic_title, label, labels, bins, fig, ax, pdf_plots):
    """

    :param arr_nn_pred:
    :param is_ic:
    :param ic_title:
    :param label:
    :param labels:
    :param dir_bins:
    :param fig:
    :param ax:
    :param pdf_plots:
    :param modelname:
    :return:
    """
    arr_nn_pred = arr_nn_pred[is_ic] # select only events with this interaction channel

    if label == 'azimuth' or label == 'zenith':
        dir_mc_index_range = (labels['dir_x']['mc_index'], labels['dir_z']['mc_index'] + 1)
        dir_pred_index_range = (labels['dir_x']['reco_index'],  labels['dir_z']['reco_index'] + 1)
        dir_mc_cart = arr_nn_pred[:, dir_mc_index_range[0] : dir_mc_index_range[1]] # cartesian
        dir_pred_cart = arr_nn_pred[:, dir_pred_index_range[0]: dir_pred_index_range[1]]

        if label == 'azimuth':
            axis_label_info = ('azimuth', 'rad')
            # atan2(y,x)
            label_mc = np.arctan2(dir_mc_cart[:, 1], dir_mc_cart[:, 0])
            label_pred = np.arctan2(dir_pred_cart[:, 1], dir_pred_cart[:, 0])

            dx_pred, dy_pred = np.clip(dir_pred_cart[:, 0], -0.999, 0.999), np.clip(dir_pred_cart[:, 1], -0.999, 0.999)

            correction = 1.253
            dx_std_pred, dy_std_pred = arr_nn_pred[:, labels['dir_x']['reco_index_std']], arr_nn_pred[:, labels['dir_y']['reco_index_std']]
            dx_std_pred, dy_std_pred = np.clip(dx_std_pred, 0, None) * correction, np.clip(dy_std_pred, 0, None) * correction

            # error propagation of atan2 with correlations between y and x neglected (covariance term = 0)
            label_pred_std = np.sqrt((dx_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dy_std_pred ** 2 +
                                     (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dx_std_pred ** 2)
            label_pred_std = np.clip(label_pred_std, None, math.pi)

        elif label == 'zenith':
            axis_label_info = ('zenith', 'rad')

            # zenith = atan2(z, sqrt(x**2 + y**2))
            label_mc = np.arctan2(dir_mc_cart[:, 2], np.sqrt(np.power(dir_mc_cart[:, 0], 2) + np.power(dir_mc_cart[:, 1], 2)))
            label_pred = np.arctan2(dir_pred_cart[:, 2], np.sqrt(np.power(dir_pred_cart[:, 0], 2) + np.power(dir_pred_cart[:, 1], 2)))

            correction = 1.253
            dz_std_pred = arr_nn_pred[:, labels['dir_z']['reco_index_std']]
            dz_std_pred = np.clip(dz_std_pred, 0, None) * correction
            dz_pred = np.clip(dir_pred_cart[:, 2], -0.999, 0.999) # == dz_pred

            label_pred_std = np.sqrt((-1 / np.sqrt((1 - dz_pred ** 2))) ** 2 * dz_std_pred ** 2) # reco std in zenith

    else:
        axis_label_info = labels[label]['str']

        label_mc = arr_nn_pred[:, labels[label]['mc_index']]
        label_pred = arr_nn_pred[:, labels[label]['reco_index']]
        label_pred_std = arr_nn_pred[:, labels[label]['index_label_std']]

    percentage_of_evts = [1.0, 0.8, 0.5, 0.2]

    cbar_max = None
    for i, percentage in enumerate(percentage_of_evts):

        n_total_evt = label_pred_std.shape[0]
        n_events_to_keep = int(n_total_evt * percentage) # how many events should be kept

        if label == 'energy': # TODO needs to be improved, maybe fit line through energyres per escale and divide by this func?
            std_pred_div_by_e_reco = np.divide(label_pred_std, label_pred)
            indices_events_to_keep = sorted(std_pred_div_by_e_reco.argsort()[:n_events_to_keep])  # select n minimum values in array

        else:
            indices_events_to_keep = sorted(label_pred_std.argsort()[:n_events_to_keep]) # select n minimum values in array

        hist_2d_label = np.histogram2d(label_mc[indices_events_to_keep], label_pred[indices_events_to_keep], bins)
        bin_edges_label = hist_2d_label[1]

        if i == 0: cbar_max = hist_2d_label[0].T.max()
        label_corr = ax.pcolormesh(bin_edges_label, bin_edges_label, hist_2d_label[0].T,
                                 norm=mpl.colors.LogNorm(vmin=1, vmax=cbar_max))

        plot_line_through_the_origin(label)

        title = plt.title('OrcaNet: ' + ic_title + ', ' + str(int(percentage * 100)) + '% of total events')
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(label_corr, ax=ax)
        cbar.ax.set_ylabel('Number of events')

        ax.set_xlabel('True ' + axis_label_info[0] + ' [' + axis_label_info[1] + ']')
        ax.set_ylabel('Reconstructed ' + axis_label_info[0] + ' [' + axis_label_info[1] + ']')
        plt.xlim(bins[0], bins[-1])
        plt.ylim(bins[0], bins[-1])

        #plt.tight_layout()

        pdf_plots.savefig(fig)

        if i > 0:
            # 1st plot, plot transparency of 100%
            hist_2d_label_all = np.histogram2d(label_mc, label_pred, bins)
            bin_edges_label_all = hist_2d_label_all[1]

            # only plot the bins that are not anymore in the histogram from above
            hist_2d_label_larger_zero = np.invert(hist_2d_label[0] == 0)
            hist_2d_label_all[0][hist_2d_label_larger_zero] = 0 # set everything to 0, when these bins are still > 0 in hist_2d_labels

            ax.pcolormesh(bin_edges_label_all, bin_edges_label_all, hist_2d_label_all[0].T, alpha=0.5,
                          norm=mpl.colors.LogNorm(vmin=1, vmax=cbar_max))
            pdf_plots.savefig(fig)
            cbar.remove()
            plt.cla()

            # 2nd plot
            # only divide nonzero bins of hist_2d_label_all
            hist_2d_label_all = np.histogram2d(label_mc, label_pred, bins)
            non_zero = hist_2d_label_all[0] > 0
            hist_2d_all_div_leftover = np.copy(hist_2d_label_all[0])
            hist_2d_all_div_leftover[non_zero] = np.divide(hist_2d_label[0][non_zero], hist_2d_label_all[0][non_zero])
            #hist_2d_all_div_leftover[np.invert(non_zero)] = -1
            corr_all_div_leftover = ax.pcolormesh(bin_edges_label_all, bin_edges_label_all, hist_2d_all_div_leftover.T, vmin=0, vmax=1)
            cbar_2 = fig.colorbar(corr_all_div_leftover, ax=ax)
            cbar_2.ax.set_ylabel('Fraction of leftover events')

            title = plt.title('OrcaNet: ' + ic_title + ', ' + str(int(percentage * 100)) + '% of total events')
            title.set_position([.5, 1.04])
            ax.set_xlabel('True ' + axis_label_info[0] + ' [' + axis_label_info[1] + ']')
            ax.set_ylabel('Reconstructed ' + axis_label_info[0] + ' [' + axis_label_info[1] + ']')

            pdf_plots.savefig(fig)
            cbar_2.remove()
            plt.cla()

        if i == 0: cbar.remove()
        plt.cla()


def for_jannik(arr_nn_pred, modelname, correct_energy=(False, 'median')):
    """

    :param arr_nn_pred:
    :param modelname:
    :param energy_bins:
    :param precuts:
    :param correct_energy:
    :return:
    """
    if correct_energy[0] is True:
        arr_nn_pred = correct_reco_energy(arr_nn_pred, metric=correct_energy[1])

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    # make cuts
    # dir_z_true > 0.75
    dir_z_true = arr_nn_pred[:, 8]
    arr_nn_pred = arr_nn_pred[dir_z_true > 0.75]

    #n p.abs(dir_z_true - reco_dir_z) < 0.05
    dir_z_true = arr_nn_pred[:, 8]
    reco_dir_z = arr_nn_pred[:, 16]
    arr_nn_pred = arr_nn_pred[np.abs(dir_z_true - reco_dir_z) < 0.05]

    # 15 - 40 GeV
    energy_true = arr_nn_pred[:, 4]
    arr_nn_pred = arr_nn_pred[np.logical_and(15 < energy_true, energy_true < 50)]

    # np.abs(dir_z_true - reco_dir_z) < 0.05

    # E_reco-E_true / E_true < 0.3
    energy_pred = arr_nn_pred[:, 9]
    energy_true = arr_nn_pred[:, 4]
    cond_en = np.abs((energy_pred - energy_true)) / energy_true
    arr_nn_pred = arr_nn_pred[cond_en < 0.3]

    vtx_x_true, vtx_y_true, vtx_z_true = arr_nn_pred[:, 9], arr_nn_pred[:, 10], arr_nn_pred[:, 11]
    r = np.sqrt(vtx_x_true ** 2 + vtx_y_true ** 2)

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages('results/plots/2d/energy/for_Jannik_' + modelname + '.pdf')

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        hist_2d_r_to_vtx_z_ic = np.histogram2d(r[is_ic], vtx_z_true[is_ic], 25)
        bin_edges_x_axis, bin_edges_y_axis = hist_2d_r_to_vtx_z_ic[1], hist_2d_r_to_vtx_z_ic[2]

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        r_to_vtx_z_ic = ax.pcolormesh(bin_edges_x_axis, bin_edges_y_axis, hist_2d_r_to_vtx_z_ic[0].T)

        title = plt.title(ic + ': r_true to vtx_z_true')
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(r_to_vtx_z_ic, ax=ax)
        cbar.ax.set_ylabel('Number of events')
        ax.set_xlabel('r_true'), ax.set_ylabel('vtx_z_true')
        plt.tight_layout()

        pdf_plots.savefig(fig)
        cbar.remove()
        ax.cla()

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        #hist_2d_r_to_vtx_z_ic = np.histogram2d(r[is_ic], vtx_z_true[is_ic], 50)
        #bin_edges_x_axis, bin_edges_y_axis = hist_2d_r_to_vtx_z_ic[1], hist_2d_r_to_vtx_z_ic[2]

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        vtx_z_true_hist = plt.hist(vtx_z_true[is_ic], 20)
        #r_to_vtx_z_ic = ax.pcolormesh(bin_edges_x_axis, bin_edges_y_axis, hist_2d_r_to_vtx_z_ic[0].T)

        title = plt.title(ic + ': vtx_z_true')
        title.set_position([.5, 1.04])
        ax.set_xlabel('vtx_z_true')
        plt.tight_layout()

        pdf_plots.savefig(fig)
        ax.cla()

    for ic in ic_list.keys():
        is_ic = get_boolean_interaction_channel_separation(arr_nn_pred[:, 2], arr_nn_pred[:, 3], ic)
        if bool(np.any(is_ic, axis=0)) is False: continue

        energy_pred = arr_nn_pred[:, 9]
        energy_true = arr_nn_pred[:, 4]
        cond_en = np.abs((energy_pred - energy_true)) / energy_true
        hist_2d_r_to_vtx_z_ic = np.histogram2d(cond_en[is_ic], vtx_z_true[is_ic], 15)
        bin_edges_x_axis, bin_edges_y_axis = hist_2d_r_to_vtx_z_ic[1], hist_2d_r_to_vtx_z_ic[2]

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        r_to_vtx_z_ic = ax.pcolormesh(bin_edges_x_axis, bin_edges_y_axis, hist_2d_r_to_vtx_z_ic[0].T)

        title = plt.title(ic + ': |(energy_pred - energy_true)| / energy_true to vtx_z_true')
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(r_to_vtx_z_ic, ax=ax)
        cbar.ax.set_ylabel('Number of events')
        ax.set_xlabel('|(energy_pred - energy_true)| / energy_true'), ax.set_ylabel('vtx_z_true')
        plt.tight_layout()

        pdf_plots.savefig(fig)
        cbar.remove()
        ax.cla()

    pdf_plots.close()
    plt.close()


#------------- Functions used in making Matplotlib plots -------------#