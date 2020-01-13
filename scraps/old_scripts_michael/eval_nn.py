#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making performance plots based on nn model predictions.
"""

import os
from matplotlib import use
import h5py
from orcanet_contrib.plotting.utils import get_cut_mask_for_events_that_exist_in_both_files
from orcanet_contrib.plotting.bg_classifier import make_prob_hists_bg_classifier, make_contamination_to_neutrino_efficiency_plot
from orcanet_contrib.plotting.ts_classifier import make_e_to_acc_plot_ts, make_ts_prob_hists, plot_ts_separability, make_e_to_acc_plot_with_diff
from orcanet_contrib.plotting.regression import (make_2d_prop_to_prop_plot,
                                                 make_1d_property_errors_metric_over_energy,
                                                 make_1d_reco_err_div_by_std_dev_plot,
                                                 make_1d_reco_err_to_reco_residual_plot,
                                                 make_2d_true_reco_plot_different_sigmas)
use('Agg')


def make_performance_plots(pred_filepath, dataset_modifier, plots_folder):
    """
    Function that makes plots based on the results of a hdf5 file with nn predictions (specified in pred_filepath).

    Parameters
    ----------
    pred_filepath : str
        Path to a h5 OrcaNet prediction file.
    dataset_modifier : str
        String that specifies the dataset modifier that is used for the file located at pred_filepath.
    plots_folder : str
        Path to the general plots folder in the OrcaNet dir structure.

    """
    pred_file = h5py.File(pred_filepath, 'r')
    main_perf_plots_path = plots_folder + '/pred_performance'

    if 'bg_classifier' in dataset_modifier:
        print('Generating plots for background classifier performance investigations')
        make_plots_subfolders(main_perf_plots_path, dataset_modifier)

        cuts = {'from_npy_str': 'bg_classifier'}
        pred_file_2 = h5py.File('/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_bg_classifier_2_class.h5', 'r')

        if pred_file_2 is not None:
            mask_1_in_2, mask_2_in_1_cut = get_cut_mask_for_events_that_exist_in_both_files(pred_file, pred_file_2, remove_duplicates=(True, False))
            cuts['mask_file_1'] = mask_1_in_2
            cuts['mask_file_2'] = mask_2_in_1_cut

        make_prob_hists_bg_classifier(pred_file, main_perf_plots_path + '/1d', savename_prefix='without_cut', cuts=cuts['mask_file_1'], x_ranges=((0, 1), (0.99, 1)))
        make_prob_hists_bg_classifier(pred_file, main_perf_plots_path + '/1d', savename_prefix='with_cut', cuts=cuts['mask_file_1'], x_ranges=((0, 1), (0.99, 1)))

        make_prob_hists_bg_classifier(pred_file_2, main_perf_plots_path + '/1d', bg_classes=['muon'],
                                      savename_prefix='standard_reco', cuts=cuts['mask_file_2'], x_ranges=((0, 1), (0, 0.1)), xlabel_prefix='Standard reco')

        if pred_file_2 is not None:
            make_contamination_to_neutrino_efficiency_plot(pred_file, pred_file_2, dataset_modifier, main_perf_plots_path + '/1d',
                                                           cuts=(cuts['mask_file_1'], cuts['mask_file_2']))

    elif dataset_modifier == 'ts_classifier':
        print('Generating plots for track-shower performance investigations')
        make_plots_subfolders(main_perf_plots_path, dataset_modifier)

        cuts = {'from_npy_str': 'neutrino_ts'}
        pred_file_2 = h5py.File('/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_ts_classifier.h5', 'r')

        if pred_file_2 is not None:
            mask_1_in_2, mask_2_in_1_cut = get_cut_mask_for_events_that_exist_in_both_files(pred_file, pred_file_2, remove_duplicates=(True, False))
            cuts['mask_file_1'] = mask_1_in_2
            cuts['mask_file_2'] = mask_2_in_1_cut

            make_e_to_acc_plot_with_diff(pred_file, pred_file_2, main_perf_plots_path + '/1d',
                                         cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                         prob_threshold_shower=0.5, plot_range=(1, 100))

            make_e_to_acc_plot_ts(pred_file_2, 'Classified as track', main_perf_plots_path + '/1d', cuts=cuts['mask_file_2'],
                                  prob_threshold_shower=0.5, savename_prefix='rdf_', save=True)
            plot_ts_separability(pred_file, main_perf_plots_path + '/1d', pred_file_2=pred_file_2,
                                 cuts=(cuts['mask_file_1'], cuts['mask_file_2']))

        make_e_to_acc_plot_ts(pred_file, 'Classified as track', main_perf_plots_path + '/1d', cuts=cuts['from_npy_str'],
                              prob_threshold_shower=0.5, savename_prefix='dl_', save=True)
        make_ts_prob_hists(pred_file, main_perf_plots_path + '/1d', cuts=cuts['from_npy_str'])

    elif 'regression' in dataset_modifier:
        make_plots_subfolders(main_perf_plots_path, dataset_modifier)
        cuts = {'from_npy_str': 'neutrino_regr'}
        reco_energy_correction = 'median'
        pred_file_2 = h5py.File('/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_regression.h5', 'r')

        if pred_file_2 is not None:
            mask_1_in_2, mask_2_in_1_cut = get_cut_mask_for_events_that_exist_in_both_files(pred_file, pred_file_2, remove_duplicates=(True, False))
            cuts['mask_file_1'] = mask_1_in_2
            cuts['mask_file_2'] = mask_2_in_1_cut

        if 'energy' in dataset_modifier:
            print('Generating plots for energy performance investigations')
            make_2d_prop_to_prop_plot(pred_file, 'energy_true', 'energy_reco', main_perf_plots_path + '/2d/energy',
                                      'e_true_to_e_reco', reco_energy_correction=reco_energy_correction,
                                      cuts=cuts['mask_file_1'], title_prefix='OrcaNet: ')
            if pred_file_2 is not None:
                make_2d_prop_to_prop_plot(pred_file_2, 'energy_true', 'energy_reco', main_perf_plots_path + '/2d/energy',
                                          'standard_reco_e_true_to_e_reco', cuts=cuts['mask_file_2'], title_prefix='Standard Reco: ')

            make_1d_property_errors_metric_over_energy(pred_file, 'energy', (None, 'median_relative'), main_perf_plots_path + '/1d/energy',
                                                       'e_true_to_median_error_energy', reco_energy_correction=reco_energy_correction,
                                                       cuts=(cuts['mask_file_1'], cuts['mask_file_2']), compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'energy', ('rel_std_div', None), main_perf_plots_path + '/1d/energy',
                                                       'e_true_to_rel_std_div_energy', reco_energy_correction=reco_energy_correction,
                                                       cuts=(cuts['mask_file_1'], cuts['mask_file_2']), compare_2nd_reco=pred_file_2)

        if 'dir' in dataset_modifier:
            print('Generating plots for directional performance investigations')
            make_2d_prop_to_prop_plot(pred_file, 'azimuth_true', 'azimuth_reco', main_perf_plots_path + '/2d/dir',
                                      'azimuth_true_to_azimuth_reco', cuts=cuts['mask_file_1'], title_prefix='OrcaNet: ')
            make_2d_prop_to_prop_plot(pred_file, 'zenith_true', 'zenith_reco', main_perf_plots_path + '/2d/dir',
                                      'zenith_true_to_zenith_reco', cuts=cuts['mask_file_1'], title_prefix='OrcaNet: ')

            if pred_file_2 is not None:
                make_2d_prop_to_prop_plot(pred_file_2, 'azimuth_true', 'azimuth_reco', main_perf_plots_path + '/2d/dir',
                                          'standard_reco_azimuth_true_to_azimuth_reco', cuts=cuts['mask_file_2'], title_prefix='Standard Reco: ')
                make_2d_prop_to_prop_plot(pred_file_2, 'zenith_true', 'zenith_reco', main_perf_plots_path + '/2d/dir',
                                          'standard_reco_zenith_true_to_zenith_reco', cuts=cuts['mask_file_2'], title_prefix='Standard Reco: ')

            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_vector', (None, 'median'), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_median_error_dirs_vect', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_spherical', (None, 'median'), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_median_error_dirs_spherical', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_spherical', (None, 'median_relative'), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_median_relative_error_dirs_spherical', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_spherical_for_experts', (None, 'median'), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_median_error_dirs_spherical_az_corr', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_spherical_for_expert_experts', (None, 'median'), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_median_error_dirs_spherical_space_angle', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_spherical', ('rel_std_div', None), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_rel_std_div_dirs_spherical', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'dirs_spherical', ('std_div', None), main_perf_plots_path + '/1d/dir',
                                                       'e_true_to_std_div_dirs_spherical', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)

        if 'bjorken' in dataset_modifier:
            print('Generating plots for bjorkeny performance investigations')
            make_2d_prop_to_prop_plot(pred_file, 'bjorkeny_true', 'bjorkeny_reco', main_perf_plots_path + '/2d/bjorkeny',
                                      'bjorkeny_true_to_bjorkeny_reco', cuts=cuts['mask_file_1'], title_prefix='OrcaNet: ')
            if pred_file_2 is not None:
                make_2d_prop_to_prop_plot(pred_file_2, 'bjorkeny_true', 'bjorkeny_reco', main_perf_plots_path + '/2d/bjorkeny',
                                          'standard_reco_bjorkeny_true_to_bjorkeny_reco', cuts=cuts['mask_file_2'], title_prefix='Standard Reco: ')

            make_1d_property_errors_metric_over_energy(pred_file, 'bjorkeny', (None, 'median'), main_perf_plots_path + '/1d/bjorkeny',
                                                       'e_true_to_median_error_bjorkeny', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)

        if 'vtx' in dataset_modifier:
            print('Generating plots for vertex performance investigations')

            make_2d_prop_to_prop_plot(pred_file, 'vtx_long_reco_mc', 'vtx_perp_reco_mc', main_perf_plots_path + '/2d/vtx',
                                      'vtx_long_reco_mc_to_vtx_perp_reco_mc', cuts=cuts['mask_file_1'], title_prefix='OrcaNet: ')

            if pred_file_2 is not None:
                make_2d_prop_to_prop_plot(pred_file_2, 'vtx_long_reco_mc', 'vtx_perp_reco_mc', main_perf_plots_path + '/2d/vtx',
                                          'vtx_long_reco_mc_to_vtx_perp_reco_mc_std', cuts=cuts['mask_file_2'],
                                          title_prefix='Standard Reco: ')

            vtx_tuples = [('vtx_x_true', 'vtx_x_reco'), ('vtx_y_true', 'vtx_y_reco'), ('vtx_z_true', 'vtx_z_reco')]

            for vtx_tpl in vtx_tuples:
                make_2d_prop_to_prop_plot(pred_file, vtx_tpl[0], vtx_tpl[1], main_perf_plots_path + '/2d/vtx',
                                          vtx_tpl[0] + '_to_' + vtx_tpl[1], cuts=cuts['mask_file_1'], title_prefix='OrcaNet: ')
                if pred_file_2 is not None:
                    make_2d_prop_to_prop_plot(pred_file_2, vtx_tpl[0], vtx_tpl[1], main_perf_plots_path + '/2d/vtx',
                                              'standard_reco_' + vtx_tpl[0] + '_to_' + vtx_tpl[1],
                                              cuts=cuts['mask_file_2'], title_prefix='Standard Reco: ')

            make_1d_property_errors_metric_over_energy(pred_file, 'vertex_vector', (None, 'median'), main_perf_plots_path + '/1d/vtx',
                                                       'e_true_to_median_error_vertex', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'vertex_vector', (None, 'median_relative'), main_perf_plots_path + '/1d/vtx',
                                                       'e_true_to_median_relative_error_vertex', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'vertex_vector', ('rel_std_div', None), main_perf_plots_path + '/1d/vtx',
                                                       'e_true_to_rel_std_div_vertex', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)
            make_1d_property_errors_metric_over_energy(pred_file, 'vertex_vector', ('std_div', None), main_perf_plots_path + '/1d/vtx',
                                                       'e_true_to_std_div_vertex', cuts=(cuts['mask_file_1'], cuts['mask_file_2']),
                                                       compare_2nd_reco=pred_file_2)

        if 'errors' in dataset_modifier:
            print('Generating plots for error performance investigations')
            make_1d_reco_err_div_by_std_dev_plot(pred_file, main_perf_plots_path + '/1d/errors', 'reco_err_div_by_std_dev', cuts=None)
            make_1d_reco_err_to_reco_residual_plot(pred_file, main_perf_plots_path + '/1d/errors', 'reco_err_to_true_reco_residual', cuts=None)
            make_2d_true_reco_plot_different_sigmas(pred_file, main_perf_plots_path + '/2d/errors', 'true_reco_plot_different_sigmas', cuts=None)

    else:
        raise NameError('The dataset_modifier ' + str(dataset_modifier) + ' is not known.')

    pred_file.close()


def make_plots_subfolders(main_perf_plots_path, dataset_modifier):
    """
    Makes the directories for the plots of a certain class_type based on the main plots dir.

    Parameters
    ----------
    main_perf_plots_path : str
        Path to the pred_performance plots folder in the OrcaNet dir structure.
    dataset_modifier : str
        String that specifies the dataset modifier.

    """
    if 'bg_classifier' in dataset_modifier:
        subfolders = ['1d']
    elif 'ts_classifier' in dataset_modifier:
        subfolders = ['1d']
    elif 'regression' in dataset_modifier:
        subfolders = ['1d/energy', '1d/dir', '1d/vtx', '1d/bjorkeny', '1d/errors',
                      '2d/energy', '2d/dir', '2d/vtx', '2d/bjorkeny', '2d/errors']
    else:
        raise ValueError('The dataset_modifier ' + str(dataset_modifier) + ' is not known.')

    for folder in subfolders:
        if not os.path.exists(main_perf_plots_path + '/' + folder):
            print('Creating directory: ' + main_perf_plots_path + '/' + folder)
            os.makedirs(main_perf_plots_path + '/' + folder)
