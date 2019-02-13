#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making performance plots based on nn model predictions.
"""

import os
import matplotlib as mpl
import h5py
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
from orcanet_contrib.plotting.bg_classifier import make_prob_hists_bg_classifier


# TODO reintegrate old regression + ts plots
def make_performance_plots(pred_filepath, class_type, plots_folder):
    """
    Function that makes plots based on the results of a hdf5 file with nn predictions (specified in pred_filepath).

    Parameters
    ----------
    pred_filepath : str
        Path to a h5 OrcaNet prediction file.
    class_type : str
        TODO
    plots_folder : str
        Path to the general plots folder in the OrcaNet dir structure.

    """
    pred_file = h5py.File(pred_filepath, 'r')
    main_perf_plots_path = plots_folder + '/pred_performance'

    if class_type == 'bg_classifier':
        make_plots_subfolders(main_perf_plots_path, class_type)
        make_prob_hists_bg_classifier(pred_file, main_perf_plots_path + '/1d')

    else:
        raise ValueError('The class_type ' + str(class_type) + ' is not known.')

    pred_file.close()

    # elif class_type == 'ts_classifier':  # categorical
    #     # TODO doesnt work
    #     precuts = (False, '3-100_GeV_prod')
    #
    #     make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title='Classified as track', filename=folder_name + 'plots/ts_' + modelname,
    #                                                   precuts=precuts, corr_cut_pred_0=0.5)
    #
    #     make_prob_hists(arr_nn_pred, folder_name, modelname=modelname, precuts=precuts)
    #     make_hist_2d_property_vs_property(arr_nn_pred, folder_name, modelname, property_types=('bjorken-y', 'probability'),
    #                                       e_cut=(1, 100), precuts=precuts)
    #     calculate_and_plot_separation_pid(arr_nn_pred, folder_name, modelname, precuts=precuts)
    #
    # else:  # regression
    #     # TODO doesnt work
    #     # TODO make the shallow reco not hardcoded
    #     arr_nn_pred_shallow = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')
    #     # precuts = (True, 'regr_3-100_GeV_prod_and_1-3_GeV_prod')
    #     precuts = (False, '3-100_GeV_prod')
    #     if 'energy' in class_type:
    #         print('Generating plots for energy performance investigations')
    #
    #         # DL
    #         make_2d_energy_resolution_plot(arr_nn_pred, modelname, folder_name, precuts=precuts,
    #                                        correct_energy=(True, 'median'))
    #         make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median_relative', precuts=precuts,
    #                                                   correct_energy=(True, 'median'), compare_shallow=(True, arr_nn_pred_shallow))
    #         make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, folder_name, precuts=precuts,
    #                                            compare_shallow=(True, arr_nn_pred_shallow), correct_energy=(True, 'median'))
    #         # shallow reco
    #         make_2d_energy_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)
    #
    #     if 'dir' in class_type:
    #         print('Generating plots for directional performance investigations')
    #
    #         # DL
    #         make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', precuts=precuts,
    #                                           compare_shallow=(True, arr_nn_pred_shallow))
    #         make_2d_dir_correlation_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
    #         # shallow reco
    #         make_2d_dir_correlation_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)
    #
    #     if 'bjorken-y' in class_type:
    #         print('Generating plots for bjorken-y performance investigations')
    #
    #         # DL
    #         make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', precuts=precuts,
    #                                                 compare_shallow=(True, arr_nn_pred_shallow))
    #         make_2d_bjorken_y_resolution_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
    #         # shallow reco
    #         make_2d_bjorken_y_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)
    #
    #     if 'errors' in class_type:
    #         print('Generating plots for error performance investigations')
    #
    #         make_1d_reco_err_div_by_std_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)  # TODO take precuts from above?
    #         make_1d_reco_err_to_reco_residual_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
    #         make_2d_dir_correlation_plot_different_sigmas(arr_nn_pred, modelname, folder_name, precuts=precuts)


def make_plots_subfolders(main_perf_plots_path, class_type):
    """
    Makes the directories for the plots of a certain class_type based on the main plots dir.

    Parameters
    ----------
    main_perf_plots_path : str
        Path to the pred_performance plots folder in the OrcaNet dir structure.
    class_type : str
        TODO

    """
    if class_type == 'bg_classifier':
        subfolders = ['1d']
    else:
        raise ValueError('The class_type ' + str(class_type) + ' is not known.')

    for folder in subfolders:
        if not os.path.exists(main_perf_plots_path + '/' + folder):
            print('Creating directory: ' + main_perf_plots_path + '/' + folder)
            os.makedirs(main_perf_plots_path + '/' + folder)


# TODO not needed anymore?
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

