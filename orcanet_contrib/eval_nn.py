#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for plotting Orca evaluations.
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from orcanet_contrib.evaluation_utilities import (make_energy_to_accuracy_plot_multiple_classes,
                                                  make_prob_hists,
                                                  make_hist_2d_property_vs_property,
                                                  calculate_and_plot_separation_pid,
                                                  make_2d_energy_resolution_plot,
                                                  make_1d_energy_reco_metric_vs_energy_plot,
                                                  make_1d_energy_std_div_e_true_plot,
                                                  make_1d_dir_metric_vs_energy_plot,
                                                  make_2d_dir_correlation_plot,
                                                  make_1d_bjorken_y_metric_vs_energy_plot,
                                                  make_2d_bjorken_y_resolution_plot,
                                                  make_1d_reco_err_div_by_std_plot,
                                                  make_1d_reco_err_to_reco_residual_plot,
                                                  make_2d_dir_correlation_plot_different_sigmas)


def old(mc_info, y_true, y_pred, class_type):
    ax = np.newaxis
    if class_type[1] == 'energy_and_direction_and_bjorken-y':
        # TODO temp old 60b prod
        y_pred = np.concatenate([y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4]], axis=1)
        y_true = np.concatenate(
            [y_true['energy'], y_true['dir_x'], y_true['dir_y'], y_true['dir_z'], y_true['bjorken-y']],
            axis=1)  # dont need to save y_true err input
    elif class_type[1] == 'energy_dir_bjorken-y_errors':
        y_pred = np.concatenate(y_pred, axis=1)
        y_true = np.concatenate([y_true['e'], y_true['dir_x'], y_true['dir_y'], y_true['dir_z'], y_true['by']],
                                axis=1)  # dont need to save y_true err input
    elif class_type[1] == 'energy_dir_bjorken-y_vtx_errors':
        y_pred = np.concatenate(y_pred, axis=1)
        y_true = np.concatenate(
            [y_true['e'], y_true['dx'], y_true['dy'], y_true['dz'], y_true['by'], y_true['vx'], y_true['vy'],
             y_true['vz'], y_true['vt']],
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
    # print(y_pred.shape)
    # print(y_true.shape)
    # alle [...,ax] haben dim 64,1.
    arr_nn_pred_temp = np.concatenate([run_id[:, ax], event_id[:, ax], particle_type[:, ax], is_cc[:, ax],
                                       energy[:, ax], bjorken_y[:, ax], dir_x[:, ax], dir_y[:, ax],
                                       dir_z[:, ax], vtx_x[:, ax], vtx_y[:, ax], vtx_z[:, ax],
                                       time_residual_vtx[:, ax], y_pred, y_true], axis=1)


# TODO Does not work at all
def investigate_model_performance(cfg, model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean, arr_filename, folder_name):
    """
    Function that 1) makes predictions based on a Keras nn model and 2) investigates the performance of the model based on the predictions.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model of a neural network.
    test_files : list(([test_filepaths], test_filesize))
        List of tuples that contains the testfiles and their number of rows.
    n_bins : list(tuple(int))
        Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    batchsize : int
        Batchsize that is used for predicting.
    class_type : str
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname.
    modelname : str
        Name of the model.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/testdata.
    arr_filename : str
        Filename of the arr_nn_pred, which will be generated or loaded. It contains all predicitons of a model
        and true values, for a specific dataset.
    folder_name : str
        Name of the folder in the cnns directory in which everything will be saved.

    """
    # for layer in model.layers: # temp
    #     if 'batch_norm' in layer.name:
    #         layer.stateful = False

    arr_nn_pred = np.load(arr_filename)

    # arr_nn_pred = np.load('results/plots/saved_predictions/arr_nn_pred_' + modelname + '_final_stateful_false.npy')
    # arr_nn_pred = np.load('results/plots/saved_predictions//arr_nn_pred_model_VGG_4d_xyz-t_and_yzt-x_and_4d_xyzt_track-shower_multi_input_single_train_tight-1_tight-2_lr_0.003_tr_st_test_st_final_stateful_false_1-100GeV_precut.npy')

    if class_type == 'track-shower':  # categorical
        precuts = (False, '3-100_GeV_prod')

        make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title='Classified as track', filename=folder_name + 'plots/ts_' + modelname,
                                                      precuts=precuts, corr_cut_pred_0=0.5)

        make_prob_hists(arr_nn_pred, folder_name, modelname=modelname, precuts=precuts)
        make_hist_2d_property_vs_property(arr_nn_pred, folder_name, modelname, property_types=('bjorken-y', 'probability'),
                                          e_cut=(1, 100), precuts=precuts)
        calculate_and_plot_separation_pid(arr_nn_pred, folder_name, modelname, precuts=precuts)

    else:  # regression
        # TODO make the shallow reco not hardcoded
        arr_nn_pred_shallow = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')
        # precuts = (True, 'regr_3-100_GeV_prod_and_1-3_GeV_prod')
        precuts = (False, '3-100_GeV_prod')
        if 'energy' in class_type:
            print('Generating plots for energy performance investigations')

            # DL
            make_2d_energy_resolution_plot(arr_nn_pred, modelname, folder_name, precuts=precuts,
                                           correct_energy=(True, 'median'))
            make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median_relative', precuts=precuts,
                                                      correct_energy=(True, 'median'), compare_shallow=(True, arr_nn_pred_shallow))
            make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, folder_name, precuts=precuts,
                                               compare_shallow=(True, arr_nn_pred_shallow), correct_energy=(True, 'median'))
            # shallow reco
            make_2d_energy_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)

        if 'dir' in class_type:
            print('Generating plots for directional performance investigations')

            # DL
            make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', precuts=precuts,
                                              compare_shallow=(True, arr_nn_pred_shallow))
            make_2d_dir_correlation_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            # shallow reco
            make_2d_dir_correlation_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)

        if 'bjorken-y' in class_type:
            print('Generating plots for bjorken-y performance investigations')

            # DL
            make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', precuts=precuts,
                                                    compare_shallow=(True, arr_nn_pred_shallow))
            make_2d_bjorken_y_resolution_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            # shallow reco
            make_2d_bjorken_y_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)

        if 'errors' in class_type:
            print('Generating plots for error performance investigations')

            make_1d_reco_err_div_by_std_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)  # TODO take precuts from above?
            make_1d_reco_err_to_reco_residual_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            make_2d_dir_correlation_plot_different_sigmas(arr_nn_pred, modelname, folder_name, precuts=precuts)


def get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident=''):
    """
    Derives the name of a model based on its number of bins and the class_type tuple.
    The final modelname is defined as 'model_Nd_proj_class_type[1]'.
    E.g. 'model_3d_xyz_muon-CC_to_elec-CC'.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) of the training images. Can contain multiple n_bins tuples.
    :param str class_type: Tuple that declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param str nn_arch: String that declares which neural network model architecture is used.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies the projection type.
    :param str str_ident: Optional str identifier that gets appended to the modelname.
    :return: str modelname: Derived modelname.
    """
    modelname = 'model_' + nn_arch + '_'

    projection = ''
    for i, bins in enumerate(n_bins):

        dim = 4 - bins.count(1)
        if i > 0:
            projection += '_and_'
        projection += str(dim) + 'd_'

        if bins.count(1) == 0 and i == 0:  # for 4D input # TODO FIX BUG XYZT AFTER NAME
            if swap_4d_channels is not None:
                projection += swap_4d_channels
            else:
                projection += 'xyz-c' if bins[3] == 31 else 'xyz-t'

        else:  # 2D/3D input
            if bins[0] > 1: projection += 'x'
            if bins[1] > 1: projection += 'y'
            if bins[2] > 1: projection += 'z'
            if bins[3] > 1: projection += 't'

    str_ident = '_' + str_ident if str_ident is not '' else str_ident
    modelname += projection + '_' + class_type + str_ident

    return modelname

