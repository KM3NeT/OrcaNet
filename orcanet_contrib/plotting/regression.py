#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making plots for regression models (e.g. energy, dir, vtx, bjorkeny, ...).
"""

import warnings
import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from orcanet_contrib.plotting.utils import correct_reco_energy, select_ic, get_event_selection_mask


# --------------------------- Code for 2d plots --------------------------- #


def make_2d_prop_to_prop_plot(pred_file, prop_1_name, prop_2_name, savefolder, savename, reco_energy_correction=None,
                              cuts=None, title_prefix='OrcaNet: '):
    """
    Function that makes a 2d plot of property 1 vs property 2.

    The available properties are grouped in the function get_make_2d_energy_resolution_plot_properties_dict().

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    prop_1_name : str
        Name of the first property that should be plotted on the X-axis.
    prop_2_name : str
        Name of the second property that should be plotted on the Y-axis.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    savename : str
        Savename with which the plots file should be saved.
    reco_energy_correction : None/str
        If on of the props is the reco energy, specifies if the energy should be corrected or not.
        If not None, the string specifies the metric that should be used for the correction.
        Available: 'median' and 'mean'.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.
    title_prefix : str
        String that is used as a prefix for the title of the plot.

    """
    properties = get_make_2d_energy_resolution_plot_properties_dict()
    if reco_energy_correction is True:
        properties['energy_reco']['ax_label'] = 'Corrected reconstructed energy (GeV)'  # add corrected info in str

    mc_info = pred_file['mc_info']
    dset_key_prop_1, dset_key_prop_2 = properties[prop_1_name]['dset_key'], properties[prop_2_name]['dset_key']
    prop_dset_1, prop_dset_2 = pred_file[dset_key_prop_1], pred_file[dset_key_prop_2]  # e.g. y_pred dset

    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        prop_dset_1 = prop_dset_1[evt_sel_mask]
        prop_dset_2 = prop_dset_2[evt_sel_mask]

    prop_1 = get_property_info_to_plot(mc_info, prop_dset_1, properties, prop_1_name, reco_energy_correction)
    prop_2 = get_property_info_to_plot(mc_info, prop_dset_2, properties, prop_2_name, reco_energy_correction)

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(savefolder + '/' + savename + '.pdf')

    for ic in ic_list.keys():
        is_ic = select_ic(mc_info['particle_type'], mc_info['is_cc'], ic)
        if bool(np.any(is_ic, axis=0)) is False:
            continue

        bins_prop_1, bins_prop_2 = properties[prop_1_name]['bins'], properties[prop_2_name]['bins']

        hist_2d_prop_1_prop_2 = np.histogram2d(prop_1[is_ic], prop_2[is_ic], [bins_prop_1, bins_prop_2])
        bin_edges_prop_1, bin_edges_prop_2 = hist_2d_prop_1_prop_2[1], hist_2d_prop_1_prop_2[2]

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        pcm_prop_1_prop_2 = ax.pcolormesh(bin_edges_prop_1, bin_edges_prop_2, hist_2d_prop_1_prop_2[0].T,
                                          norm=mpl.colors.LogNorm(vmin=1, vmax=hist_2d_prop_1_prop_2[0].T.max()))

        title_name_prop_1, title_name_prop_2 = properties[prop_1_name]['title_name'], properties[prop_2_name]['title_name']

        plot_line_through_the_origin(prop_1_name, prop_2_name)
        title = plt.title(title_prefix + ic_list[ic]['title'] + ', ' + title_name_prop_1 + ' vs. ' + title_name_prop_2)
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(pcm_prop_1_prop_2, ax=ax)
        cbar.ax.set_ylabel('Number of events')
        x_label, y_label = properties[prop_1_name]['ax_label'], properties[prop_2_name]['ax_label']
        ax.set_xlabel(x_label), ax.set_ylabel(y_label)
        plt.tight_layout()

        pdf_plots.savefig(fig)
        cbar.remove()  # TODO still necessary?
        ax.cla()

    pdf_plots.close()
    plt.close()


def get_make_2d_energy_resolution_plot_properties_dict():
    """
    Returns a dict with a list of all properties that should work with the make_2d_energy_resolution_plot_properties
    function.

    Returns
    -------
    properties : dict
        Dict that contains the configurations of all properties that work with the
        make_2d_energy_resolution_plot_properties function.

    """
    properties = {
                  'energy_reco': {'dset_key': 'pred', 'col_name': 'pred_energy', 'bins': np.arange(1, 101, 1),
                                  'title_name': 'reco energy', 'ax_label': 'Reconstructed energy (GeV)'},
                  'energy_true': {'dset_key': 'mc_info', 'col_name': 'energy', 'bins': np.arange(1, 101, 1),
                                  'title_name': 'true energy', 'ax_label': 'True energy (GeV)'},
                  'dir_x_reco': {'dset_key': 'pred', 'col_name': 'pred_dir_x', 'bins': np.linspace(-1, 1, 100),
                                 'title_name': 'reco dir-x', 'ax_label': 'Reconstructed dir-x'},
                  'dir_y_reco': {'dset_key': 'pred', 'col_name': 'pred_dir_y', 'bins': np.linspace(-1, 1, 100),
                                 'title_name': 'reco dir-y', 'ax_label': 'Reconstructed dir-y'},
                  'dir_z_reco': {'dset_key': 'pred', 'col_name': 'pred_dir_z', 'bins': np.linspace(-1, 1, 100),
                                 'title_name': 'reco dir-z', 'ax_label': 'Reconstructed dir-z'},
                  'dir_x_true': {'dset_key': 'mc_info', 'col_name': 'dir_x', 'bins': np.linspace(-1, 1, 100),
                                 'title_name': 'true dir-x', 'ax_label': 'True dir-x'},
                  'dir_y_true': {'dset_key': 'mc_info', 'col_name': 'dir_y', 'bins': np.linspace(-1, 1, 100),
                                 'title_name': 'true dir-y', 'ax_label': 'True dir-y'},
                  'dir_z_true': {'dset_key': 'mc_info', 'col_name': 'dir_z', 'bins': np.linspace(-1, 1, 100),
                                 'title_name': 'true dir-z', 'ax_label': 'True dir-z'},
                  'azimuth_reco': {'dset_key': 'pred', 'bins': np.linspace(-math.pi, math.pi, 100),
                                   'title_name': 'reco azimuth', 'ax_label': 'Reconstructed azimuth (rad)'},
                  'azimuth_true': {'dset_key': 'mc_info', 'bins': np.linspace(-math.pi, math.pi, 100),
                                   'title_name': 'true azimuth', 'ax_label': 'True azimuth (rad)'},
                  'zenith_reco': {'dset_key': 'pred', 'bins': np.linspace(-math.pi/float(2), math.pi/float(2), 100),
                                  'title_name': 'reco zenith', 'ax_label': 'Reconstructed zenith (rad)'},
                  'zenith_true': {'dset_key': 'mc_info', 'bins': np.linspace(-math.pi/float(2), math.pi/float(2), 100),
                                  'title_name': 'true zenith', 'ax_label': 'True zenith (rad)'},
                  'bjorkeny_reco': {'dset_key': 'pred', 'col_name': 'pred_bjorkeny', 'bins': np.linspace(0, 1, 101),
                                    'title_name': 'reco bjorkeny', 'ax_label': 'Reconstructed bjorkeny'},
                  'bjorkeny_true': {'dset_key': 'mc_info', 'col_name': 'bjorkeny', 'bins': np.linspace(0, 1, 101),
                                    'title_name': 'true bjorkeny', 'ax_label': 'True bjorkeny'},
                  'vtx_x_reco': {'dset_key': 'pred', 'col_name': 'pred_vtx_x', 'bins': 50,
                                 'title_name': 'reco vtx-x', 'ax_label': 'Reconstructed vtx-x'},
                  'vtx_y_reco': {'dset_key': 'pred', 'col_name': 'pred_vtx_y', 'bins': 50,
                                 'title_name': 'reco vtx-y', 'ax_label': 'Reconstructed vtx-y'},
                  'vtx_z_reco': {'dset_key': 'pred', 'col_name': 'pred_vtx_z', 'bins': 50,
                                 'title_name': 'reco vtx-z', 'ax_label': 'Reconstructed vtx-z'},
                  'vtx_x_true': {'dset_key': 'mc_info', 'col_name': 'vtx_x', 'bins': 50,
                                 'title_name': 'true vtx-x', 'ax_label': 'True vtx-x'},
                  'vtx_y_true': {'dset_key': 'mc_info', 'col_name': 'vtx_y', 'bins': 50,
                                 'title_name': 'true vtx-y', 'ax_label': 'True vtx-y'},
                  'vtx_z_true': {'dset_key': 'mc_info', 'col_name': 'vtx_z', 'bins': 50,
                                 'title_name': 'true vtx-z', 'ax_label': 'True vtx-z'}
                  }

    return properties


def get_property_info_to_plot(mc_info, prop_dset, properties, prop_name, reco_energy_correction=None):
    """
    Returns a 1d array which contains the data of the specified property, that should be plotted on one axis.

    Parameters
    ----------
    mc_info : h5py.dataset.Dataset/ndarray(ndim=2)
        The mc_info structured array of an OrcaNet nn prediction file.
    prop_dset : h5py.dataset.Dataset/ndarray(ndim=2)
        The dataset, which contains the property as one column. Coincidentally, can also be equal to the
        mc_info dataset, e.g. if prop_name = 'energy_true'
    properties : dict
        Dict that contains the configurations of all properties that work with the
        make_2d_energy_resolution_plot_properties function.
    prop_name : str
        Name of the property.
    reco_energy_correction : None/str
        If on of the props is the reco energy, specifies if the energy should be corrected or not.
        If not None, the string specifies the metric that should be used for the correction.
        Available: 'median' and 'mean'.

    Returns
    -------
    prop : ndarray(ndim=1)
        1d array which contains the data of the specified property that will be plotted on one axis later on.

    """
    if 'azimuth' in prop_name or 'zenith' in prop_name:
        # atan2(y,x)
        dset_key = properties[prop_name]['dset_key']
        col_name_prefix = '' if dset_key != 'pred' else 'pred_'

        azimuth = convert_vectorial_to_spherical_dir(prop_dset, 'azimuth', col_name_prefix)
        prop = azimuth

    elif 'zenith' in prop_name:
        # atan2(z, sqrt(x**2 + y**2))
        dset_key = properties[prop_name]['dset_key']
        col_name_prefix = '' if dset_key != 'pred' else 'pred_'

        zenith = convert_vectorial_to_spherical_dir(prop_dset, 'zenith', col_name_prefix)
        prop = zenith

    elif 'bjorkeny_true' in prop_name:
        # correct true by to 1 for e-NC events for the plotting
        abs_particle_type, is_cc = np.abs(mc_info['particle_type']), mc_info['is_cc']
        is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)

        col_name = properties[prop_name]['col_name']  # should be bjorkeny
        # here, prop array is actually the column bjorkeny of the mc_info array
        prop = np.copy(prop_dset[col_name])  # TODO test if this really works or if this is actually necessary
        prop[is_e_nc] = 1

    elif prop_name == 'energy_reco':
        col_name = properties[prop_name]['col_name']

        if reco_energy_correction is not None:
            energy_pred_array = prop_dset[col_name]
            energy_pred = correct_reco_energy(mc_info, energy_pred_array, metric=reco_energy_correction)
            prop = energy_pred
        else:
            prop = prop_dset[col_name]

    else:
        # energy_true, dir_x, dir_y, dir_z, vtx_x, ...
        col_name = properties[prop_name]['col_name']
        prop = prop_dset[col_name]

    return prop

# --------------------------- Code for 2d plots --------------------------- #

# --------------------------- Code for 1d plots --------------------------- #


def make_1d_property_errors_metric_over_energy(pred_file, property_name, mode, savefolder, savename,
                                               reco_energy_correction=None, energy_bins=np.linspace(1, 100, 32),
                                               cuts=None, compare_2nd_reco=None):
    """
    Makes binned 1d plots that show
    1) X-axis: The true energy binned into bins
    2) Y-axis: Some performance metric, dependent on the mode[0] argument.

    If mode[0] == 'rel_std_div':
        Plots the relative standard deviation (relative to the e_mc energy bin) of the property
        specified by the property_name
    If mode[0] != 'rel_std_div'
        Plots the reco error residuals of prop_reco - prop_true, binned in energy with a certain metric
        (e.g. mae, median) for the specified interaction channel. The 2nd tuple element mode[1] is the metric
        and must be one of "mae", "median", "mean_relative" or "median_relative".

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    property_name : str
        Name of the property that should be plotted on the Y-axis, e.g. 'energy'.
        All available properties can be found in the property dict below.
    mode : tuple(str, str)
        The description of this parameter can be found above in the docs of the main function.
        If mode[0] != 'rel_std_div', mode[1] must be one of "mae", "median", "mean_relative" or "median_relative".
        In this case, what is given in mode[0] doesn't matter at all.
        Likewise, if mode[0] == 'rel_std_div', what is specified in mode[1] doesn't matter.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    savename : str
        Savename with which the plots file should be saved.
    reco_energy_correction : None/str
        If the prop is the reco energy, specifies if the energy should be corrected or not.
        If not None, the string specifies the metric that should be used for the correction.
        Available: 'median' and 'mean'.
    energy_bins : ndarray(ndim=1)
        Energy bins that should be used for the binning of the X-axis.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.
    compare_2nd_reco : None/h5py.File
        Either None or h5py.File instance, if the info of a 2nd reco should be plotted into the same plots.
        The file should have the same format as the pred_file.

    """
    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(savefolder + '/' + savename + '.pdf')
    title_prefix = 'OrcaNet: '

    properties = {'dirs_vector': {'sub_props': ['dir_x', 'dir_y', 'dir_z'], 'ylabel': ' error (dir)'},
                  'dirs_spherical': {'sub_props': ['azimuth', 'zenith'], 'ylabel': ' error (dir)'},
                  'energy': {'sub_props': ['energy'], 'ylabel': ' error (corrected energy)', 'correct': 'median'},
                  'vertex_vector': {'sub_props': ['vtx_x', 'vtx_y', 'vtx_z'], 'ylabel': ' error (vertex)'},
                  'bjorkeny': {'sub_props': ['bjorkeny'], 'ylabel': ' error (bjorkeny)'}}

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    mc_info, pred = get_mc_info_and_pred_datasets(pred_file, 'mc_info', 'pred', cuts)
    if compare_2nd_reco is not None:
        pred_file_2 = compare_2nd_reco
        mc_info_2, pred_2 = get_mc_info_and_pred_datasets(pred_file_2, 'mc_info', 'pred', cuts)

    sub_props_list = properties[property_name]['sub_props']
    for ic in ic_list.keys():
        is_ic = select_ic(mc_info['particle_type'], mc_info['is_cc'], ic)
        if bool(np.any(is_ic, axis=0)) is False:
            continue

        for sub_prop in sub_props_list:
            plot_data = calc_plot_data_of_energy_dependent_label(mc_info, pred, sub_prop, mode,
                                                                 energy_bins=energy_bins,
                                                                 reco_energy_correction=reco_energy_correction,)

            bins, perf = plot_data[0], plot_data[1]
            ax.step(bins, perf, linestyle="-", where='post', label='DL ' + sub_prop)

            if compare_2nd_reco is not None:
                # no energy correction for the 2nd file, typically some standard reco
                plot_data_2 = calc_plot_data_of_energy_dependent_label(mc_info_2, pred_2, sub_prop, mode,
                                                                       reco_energy_correction=None,
                                                                       energy_bins=energy_bins)

                ax.step(bins, plot_data_2[1], linestyle="-", where='post', label='Std ' + sub_prop)

        x_ticks_major = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks_major)
        ax.minorticks_on()
        title = plt.title(title_prefix + ic_list[ic]['title'])
        title.set_position([.5, 1.04])
        y_label = r'$\sigma / E_{true}$ for ' + property_name + ' reco' if mode[0] == 'rel_std_div' else mode[1] + properties[property_name]['ylabel']
        ax.set_xlabel('True energy (GeV)'), ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend(loc='upper right')

        pdf_plots.savefig(fig)
        ax.cla()

    plt.close()
    pdf_plots.close()


def get_mc_info_and_pred_datasets(pred_file, mc_info_key, pred_key, cuts):
    """
    Gets the mc_info and pred_datasets from a h5py pred_file instance and applies some cuts if cuts is not None.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    mc_info_key : str
        Key of the mc_info dataset in the pred_file.
    pred_key : str
        Key of the pred dataset in the pred_file.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.

    Returns
    -------
    mc_info : h5py.dataset.Dataset/ndarray(ndim=2)
        The (cut) mc_info structured array of an OrcaNet nn prediction file.
    pred : h5py.dataset.Dataset/ndarray(ndim=2)
        The (cut) pred dataset of a nn OrcaNet classifier, which contains all predicted labels as single columns.

    """
    mc_info = pred_file[mc_info_key]
    pred = pred_file[pred_key]
    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        pred = pred[evt_sel_mask]

    return mc_info, pred


def calc_plot_data_of_energy_dependent_label(mc_info, pred, prop_name, mode,
                                             energy_bins=np.linspace(1, 100, 20), reco_energy_correction=None):
    """
    Returns two different kind of performance metrics vs energy, dependent on the mode parameter:

    1) if mode[0] == 'rel_std_div':
       Gets the relative standard deviation for each bin, which are specified by the energy_bins parameter.
    2) if mode[0] != 'rel_std_div'
        Returns the reco error residuals of prop_reco - prop_true, binned in energy with a certain metric
        (e.g. mae, median). The 2nd tuple element mode[1] is the metric and must be one of
        "mae", "median", "mean_relative" or "median_relative".

    Parameters
    ----------
    mc_info : h5py.dataset.Dataset/ndarray(ndim=2)
        The mc_info structured array of an OrcaNet nn prediction file.
    pred : h5py.dataset.Dataset/ndarray(ndim=2)
        The pred dataset of a nn OrcaNet classifier, which contains all predicted labels as single columns.
    prop_name : str
        Name of the property, of which the reco error residuals should be calculated, e.g. "energy".
    mode : tuple(str, str)
        The description of this parameter can be found above in the docs of the main function.
        If mode[0] != 'rel_std_div', mode[1] must be one of "mae", "median", "mean_relative" or "median_relative".
        In this case, what is given in mode[0] doesn't matter at all.
        Likewise, if mode[0] == 'rel_std_div', what is specified in mode[1] doesn't matter.
    energy_bins : ndarray(ndim=1)
        Energy bins that should be used for the binning.
    reco_energy_correction : None/str
        If the prop is the reco energy, specifies if the energy should be corrected or not.
        If not None, the string specifies the metric that should be used for the correction.
        Available: 'median' and 'mean'.

    Returns
    -------
    energy_to_property_performance_plot_data_ic : tuple(ndarray, ndarray, optional(ndarray))
        If mode[0] == 'rel_std_div':
            Tuple containing 2 arrays:
            1) energy_bins array from the input.
            2) the relative (relative to e-mc-bin) standard deviation of the property for each bin.
        If mode[0] != 'rel_std_div':
            Tuple containing 2, or in case of 'median_relative' 3, arrays:
            1) energy_bins array from the input.
            2) hist_energy_residuals_metric array, which contains the metric value for each energy bin.
            3) hist_energy_variance array, same as 2 but this time the variance.

    """
    if 'azimuth' in prop_name:
        # atan2(y,x)
        azimuth_pred = convert_vectorial_to_spherical_dir(pred, 'azimuth', 'pred_')
        azimuth_true = convert_vectorial_to_spherical_dir(mc_info, 'azimuth', 'pred_')

        prop_pred, prop_true = azimuth_pred, azimuth_true

    elif 'zenith' in prop_name:
        # atan2(z, sqrt(x**2 + y**2))
        zenith_pred = convert_vectorial_to_spherical_dir(pred, 'zenith', 'pred_')
        zenith_true = convert_vectorial_to_spherical_dir(mc_info, 'zenith', 'pred_')

        prop_pred, prop_true = zenith_pred, zenith_true

    elif 'energy' in prop_name:
        if reco_energy_correction is not None:
            energy_pred_array = pred['pred_energy']
            energy_pred = correct_reco_energy(mc_info, energy_pred_array, metric=reco_energy_correction)
            energy_true = mc_info[prop_name]
            prop_pred, prop_true = energy_pred, energy_true
        else:
            prop_pred, prop_true = pred['pred_' + prop_name], mc_info[prop_name]

    elif 'bjorkeny' in prop_name:
        # correct by to 1 for e-NC events
        abs_particle_type, is_cc = np.abs(mc_info['particle_type']), mc_info['is_cc']
        is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)

        bjorkeny_pred = np.copy(pred['pred_bjorkeny'])  # TODO test if this really works
        bjorkeny_pred[is_e_nc] = 1

        bjorkeny_true = mc_info['bjorkeny']

        prop_pred, prop_true = bjorkeny_pred, bjorkeny_true

    else:
        prop_pred, prop_true = pred['pred_' + prop_name], mc_info[prop_name]

    energy_true = mc_info['energy']
    if mode[0] == 'rel_std_div':
        energy_to_property_performance_plot_data_ic = get_rel_std_div_plot_data(prop_pred, energy_true, energy_bins)
    else:
        metric = mode[1]
        err = np.abs(prop_pred - prop_true)
        energy_to_property_performance_plot_data_ic = bin_error_in_energy_bins(err, energy_true, energy_bins, metric=metric)

    return energy_to_property_performance_plot_data_ic


def get_rel_std_div_plot_data(prop_pred, energy_true, energy_bins):
    """
    Function that calculates the relative standard deviation of prop_pred in a certain energy range and returns
    it together with the energy_bins array from the input.

    Parameters
    ----------
    prop_pred : ndarray(ndim=1)
        Prediction values of a property like e.g. energy.
    energy_true : ndarray(ndim=1)
        True MC energy values.
    energy_bins : ndarray(ndim=1)
        Array which specifies the energy bins that should be used for the x-binning.

    Returns
    -------
    energy_binned_rel_std_plot_data : tuple(ndarray, ndarray)
        Tuple containing 2 arrays:
        1) energy_bins array from the input.
        2) the relative (relative to e-mc-bin) standard deviation of the property for each bin.

    """
    std_rel = []  # y-axis data
    for i in range(energy_bins.shape[0] - 1):
        e_range_low, e_range_high = energy_bins[i], energy_bins[i+1]
        e_range_mean = (e_range_low + e_range_high) / float(2)

        # cut out prop_pred values within a certain energy bin
        prop_pred_cut_boolean = np.logical_and(e_range_low < energy_true, energy_true <= e_range_high)
        prop_pred_cut = prop_pred[prop_pred_cut_boolean]

        std_temp = np.std(prop_pred_cut)
        std_rel.append(std_temp / float(e_range_mean))

    # fix for mpl
    std_rel.append(std_rel[-1])
    energy_binned_rel_std_plot_data = [energy_bins, std_rel]

    return energy_binned_rel_std_plot_data


def bin_error_in_energy_bins(err, mc_energy, energy_bins, metric='median_relative'):
    """
    Bins the err residuals into energy bins, and calculates the value of each bin based on a specified metric.

    Parameters
    ----------
    err : ndarray(ndim=1)
        Array with the absolute residual reco - true.
    mc_energy : ndarray(ndim=1)
        Array with the true mc energy.
    energy_bins : ndarray(ndim=1)
        Array which specifies the energy bins that should be used for the x-binning.
    metric : str
        Metric that should be used for calculating the values for each bin.
        One of "mae", "median", "mean_relative", "median_relative".

    Returns
    -------
    energy_binned_err_plot_data : tuple(ndarray, ndarray, optional(ndarray))
        Tuple containing 2, or in case of 'median_relative' or 'mean_relative' 3, arrays:
        1) energy_bins array from the input.
        2) hist_energy_residuals_metric array, which contains the metric value for each energy bin.
        3) hist_energy_variance array, same as 2 but this time the variance.

    """
    # bin the err, depending on their mc_energy, into energy_bins
    hist_energy_residuals_metric = np.zeros((len(energy_bins) - 1))
    hist_energy_variance = np.zeros((len(energy_bins) - 1))
    # In which bin does each event belong, according to its mc energy:
    bin_indices = np.digitize(mc_energy, bins=energy_bins)

    # For every mc energy bin, calculate the merge metric (e.g. mae or median relative) of all events that have a corresponding mc energy
    for bin_no in range(1, len(energy_bins)):
        current_err = err[bin_indices == bin_no]
        current_mc_energy = mc_energy[bin_indices == bin_no]

        if metric == 'mae':
            # calculate mean absolute error
            hist_energy_residuals_metric[bin_no - 1] = np.mean(current_err)
        elif metric == 'median_relative':
            # calculate the median of the relative error: |label_reco-label_true|/E_true and also its variance
            # probably makes only sense if the label from the err array is the energy
            relative_error = current_err / current_mc_energy
            hist_energy_residuals_metric[bin_no - 1] = np.median(relative_error)
            hist_energy_variance[bin_no - 1] = np.var(relative_error)
        elif metric == 'mean_relative':
            # calculate the median of the relative error: |label_reco-label_true|/E_true and also its variance
            # probably makes only sense if the label from the err array is the energy
            relative_error = current_err / current_mc_energy
            hist_energy_residuals_metric[bin_no - 1] = np.mean(relative_error)
            hist_energy_variance[bin_no - 1] = np.var(relative_error)
        elif metric == 'median':
            hist_energy_residuals_metric[bin_no - 1] = np.median(current_err)
        else:
            raise ValueError('Operation modes other than "mae" and "median_relative" are not supported.')

    # For proper plotting with plt.step and where="post"
    hist_energy_residuals_metric = np.append(hist_energy_residuals_metric, hist_energy_residuals_metric[-1])
    hist_energy_variance = np.append(hist_energy_variance, hist_energy_variance[-1]) if metric in ['median_relative', 'mean_relative'] else None
    energy_binned_err_plot_data = [energy_bins, hist_energy_residuals_metric, hist_energy_variance]

    return energy_binned_err_plot_data

# --------------------------- Code for 1d plots --------------------------- #


# --------------------------- Code for error plots --------------------------- #


def make_1d_reco_err_div_by_std_dev_plot(pred_file, savefolder, savename, cuts=None):
    """
    Makes a 1d hist plot which shows the reco err (y_true - y_reco) divided
    by the predicted standard deviation error of a nn.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    savename : str
        Savename with which the plots file should be saved.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.

    """
    # do this for all labels: 1d (y_true - y_reco) / std_reco
    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(savefolder + '/' + savename + '.pdf')

    prop_names = {'energy': {'col_name_pred': 'pred_energy', 'col_name_pred_err': 'pred_err_energy', 'col_name_true': 'true_energy'},
                  'bjorkeny': {'col_name_pred': 'pred_bjorkeny', 'col_name_pred_err': 'pred_err_bjorkeny', 'col_name_true': 'true_bjorkeny'},
                  'dir_x': {'col_name_pred': 'pred_dir_x', 'col_name_pred_err': 'pred_err_dir_x', 'col_name_true': 'true_dir_x'},
                  'dir_y': {'col_name_pred': 'pred_dir_y', 'col_name_pred_err': 'pred_err_dir_y', 'col_name_true': 'true_dir_y'},
                  'dir_z': {'col_name_pred': 'pred_dir_z', 'col_name_pred_err': 'pred_err_dir_z', 'col_name_true': 'true_dir_z'},
                  'vtx_x': {'col_name_pred': 'pred_vtx_x', 'col_name_pred_err': 'pred_err_vtx_x', 'col_name_true': 'true_vxt_x'},
                  'vtx_y': {'col_name_pred': 'pred_vtx_y', 'col_name_pred_err': 'pred_err_vtx_y', 'col_name_true': 'true_vxt_y'},
                  'vtx_z': {'col_name_pred': 'pred_vtx_z', 'col_name_pred_err': 'pred_err_vtx_z', 'col_name_true': 'true_vxt_z'},
                  'vtx_t': {'col_name_pred': 'pred_vtx_t', 'col_name_pred_err': 'pred_err_vtx_t', 'col_name_true': 'true_vxt_t'}}

    mc_info = pred_file['mc_info']
    for prop in prop_names:
        prop_pred = pred_file['pred'][prop_names[prop]['col_name_pred']]
        prop_pred_err = pred_file['pred'][prop_names[prop]['col_name_pred_err']]
        prop_true = pred_file['true'][prop_names[prop]['col_name_true']]

        if cuts is not None:
            assert isinstance(cuts, str)
            evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
            prop_pred = prop_pred[evt_sel_mask]
            prop_pred_err = prop_pred_err[evt_sel_mask]
            prop_true = prop_true[evt_sel_mask]

        plot_1d_reco_err_div_by_std_for_label(prop_pred, prop_pred_err, prop_true, fig, ax, pdf_plots, prop)

    plt.close()
    pdf_plots.close()


def plot_1d_reco_err_div_by_std_for_label(prop_pred, prop_pred_err, prop_true, fig, ax, pdf_plots, label):
    """
    Plots the reco err (y_true - y_reco) divided by the predicted standard deviation error
    to an ax of a fig and saves them to the pdf_plots instance.

    Parameters
    ----------
    prop_pred : ndarray(ndim=1)
        Array with the nn predictions of a certain property (e.g. pred_energy).
    prop_pred_err : ndarray(ndim=1)
        Array with the nn error predictions of a certain property.
    prop_true : ndarray(ndim=1)
        Array with the true values of a certain property (e.g. pred_energy).
    fig : mpl figure
        Matplotlib instance of a figure.
    ax : mpl.axes
        Axes object that refers to ax of an existing plt.sublots object.
    pdf_plots : mpl.backends.backend_pdf.PdfPages
        Instance of a matplotlib PdfPages object, to which the plot should be saved.
    label : str
        Label that should be used as a str in the plot with plt.hist.

    """
    # convert mse error to standard deviation with the magic number sqrt(pi/2)
    prop_pred_err_std = prop_pred_err * 1.253

    pred_err_div_by_std = np.divide(prop_true - prop_pred, np.abs(prop_pred_err_std))
    exclude_outliers = np.logical_and(pred_err_div_by_std > -10, pred_err_div_by_std < 10)

    # print(np.std(pred_err_div_by_std[exclude_outliers]))
    plt.hist(pred_err_div_by_std[exclude_outliers], bins=100, label=label)

    title = plt.title('Gaussian Likelihood errors for ' + label)
    title.set_position([.5, 1.04])
    ax.set_xlabel(r'$(y_{\mathrm{true}} - y_{\mathrm{pred}})/ \sigma_{\mathrm{pred}}$'), ax.set_ylabel('Counts [#]')
    ax.grid(True)

    pdf_plots.savefig(fig)
    ax.cla()


def make_1d_reco_err_to_reco_residual_plot(pred_file, savefolder, savename, cuts=None):
    """
    Makes a 1d plot which shows the true standard deviation of the error residuals (abs(true-pred)) versus
    the estimated uncertainty sigma_pred by a nn.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    savename : str
        Savename with which the plots file should be saved.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.

    """
    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(savefolder + '/' + savename + '.pdf')

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    properties = {'energy': {'col_name_pred': 'pred_energy', 'col_name_true': 'true_energy', 'col_name_pred_err': 'pred_err_energy', 'unit': '[GeV]'},
                  'bjorkeny': {'col_name_pred': 'pred_bjorkeny', 'col_name_true': 'true_bjorkeny', 'col_name_pred_err': 'pred_err_bjorkeny', 'unit': ''},
                  'dir_x': {'col_name_pred': 'pred_dir_x', 'col_name_true': 'true_dir_x', 'col_name_pred_err': 'pred_err_dir_x', 'unit': '[rad]'},
                  'dir_y': {'col_name_pred': 'pred_dir_y', 'col_name_true': 'true_dir_y', 'col_name_pred_err': 'pred_err_dir_y', 'unit': '[rad]'},
                  'dir_z': {'col_name_pred': 'pred_dir_z', 'col_name_true': 'true_dir_z', 'col_name_pred_err': 'pred_err_dir_z', 'unit': '[rad]'},
                  'azimuth': {'col_name_pred': None, 'col_name_true': None, 'col_name_pred_err': None, 'unit': '[rad]'},
                  'zenith': {'col_name_pred': None, 'col_name_true': None, 'col_name_pred_err': None, 'unit': '[rad]'},
                  'vtx_x': {'col_name_pred': 'pred_vtx_x', 'col_name_true': 'true_vtx_x', 'col_name_pred_err': 'pred_err_vtx_x', 'unit': '[rad]'},
                  'vtx_y': {'col_name_pred': 'pred_vtx_y', 'col_name_true': 'true_vtx_y', 'col_name_pred_err': 'pred_err_vtx_y', 'unit': '[rad]'},
                  'vtx_z': {'col_name_pred': 'pred_vtx_z', 'col_name_true': 'true_vtx_z', 'col_name_pred_err': 'pred_err_vtx_z', 'unit': '[rad]'},
                  'vtx_t': {'col_name_pred': 'pred_vtx_t', 'col_name_true': 'true_vtx_t', 'col_name_pred_err': 'pred_err_vtx_t', 'unit': '[rad]'}}

    mc_info, pred, true = pred_file['mc_info'], pred_file['pred'], pred_file['true']
    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        pred = pred[evt_sel_mask]
        true = true[evt_sel_mask]

    for prop_name in properties:
        for ic in ic_list:
            is_ic = select_ic(mc_info['particle_type'], mc_info['is_cc'], ic)
            if bool(np.any(is_ic, axis=0)) is False:
                continue

            mc_info_ic = mc_info[is_ic]
            pred_ic, true_ic = pred[is_ic], true[is_ic]
            prop_true, prop_pred, prop_pred_err = get_prop_true_pred_and_prop_pred_sigma(pred_ic, true_ic, mc_info_ic,
                                                                                         properties, prop_name)

            n_x_bins = 50
            title = plt.title(ic_list[ic]['title'] + ': ' + prop_name)
            unit = properties[prop_name]['unit']

            plot_1d_reco_err_to_reco_residual_for_prop(prop_true, prop_pred, prop_pred_err, n_x_bins, fig,
                                                       ax, title, pdf_plots, prop_name, unit)

    plt.close()
    pdf_plots.close()


def get_prop_true_pred_and_prop_pred_sigma(pred, true, mc_info, properties, prop_name):
    """
    Gets the data that is needed to plot the true standard deviation of the error residuals (abs(true-pred)) versus
    the estimated uncertainty sigma_pred by a nn.

    Parameters
    ----------
    pred : h5py.dataset.Dataset/ndarray(ndim=2)
        The pred dataset of an OrcaNet nn prediction file.
    true : h5py.dataset.Dataset/ndarray(ndim=2)
        The true dataset of an OrcaNet nn prediction file.
    mc_info : h5py.dataset.Dataset/ndarray(ndim=2)
        The mc_info dataset of an OrcaNet nn prediction file.
    properties : dict
        Dict which contains some configurations of all properties.
    prop_name : str
        String which specifies the specific property that should be used.

    Returns
    -------
    prop_true : ndarray(ndim=1)
        Array which contains the true values of the specified property.
    prop_pred : ndarray(ndim=1)
        Array which contains the predicted values of the specified property.
    prop_pred_err : ndarray(ndim=1)
        Array which contains the predicted error values of the specified property, multiplied by the magic number
        sqrt(pi/2) in order to get a standard deviation from the error values if trained with mse loss.
        If the prop is azimuth or zenith, the error of the vectorial dirs is propagated with error propagation
        that neglects the correlation term.

    """

    if prop_name == 'azimuth' or prop_name == 'zenith':

        dx_true, dx_pred = true[properties['dir_x']['col_name_true']], pred[properties['dir_x']['col_name_pred']]
        dy_true, dy_pred = true[properties['dir_y']['col_name_true']], pred[properties['dir_y']['col_name_pred']]
        dz_true, dz_pred = true[properties['dir_z']['col_name_true']], pred[properties['dir_z']['col_name_pred']]

        dx_pred_err = pred[properties['dir_x']['col_name_pred_err']]
        dy_pred_err = pred[properties['dir_y']['col_name_pred_err']]
        dz_pred_err = pred[properties['dir_z']['col_name_pred_err']]

        correction = 1.253  # convert mse error to standard deviation with the magic number sqrt(pi/2)
        if prop_name == 'azimuth':
            azimuth_true = convert_vectorial_to_spherical_dir(true, 'azimuth', col_name_prefix='')
            azimuth_pred = convert_vectorial_to_spherical_dir(pred, 'azimuth', col_name_prefix='pred_')

            # clip std prediction to 0 (maybe not really necessary) and apply correction
            dx_pred_err = np.clip(np.copy(dx_pred_err), 0, None) * correction  # copy maybe not necessary
            dy_pred_err = np.clip(np.copy(dy_pred_err), 0, None) * correction

            # error propagation of atan2 with correlations between y and x neglected (covariance term = 0)
            azimuth_pred_err = np.sqrt((dx_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dy_pred_err ** 2 +
                                       (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dx_pred_err ** 2)

            # clip every predicted standard deviation to pi if larger than pi
            n = azimuth_pred_err > math.pi
            percentage_true = np.sum(n) / float(azimuth_pred_err.shape[0]) * 100
            print('Clipped ' + str(n) + ' predicted azimuth standard deviations to pi ('
                  + str(percentage_true) + '% of all events)')

            azimuth_pred_err = np.clip(azimuth_pred_err, None, math.pi)

            # # If we don't want to neglect the convariance term , since y and x are correlated:
            # # tested: doesn't really make a difference
            # mean_dx, mean_dy = np.mean(dx_pred), np.mean(dy_pred)
            # covariance_xy = 1/float(dx_pred.shape[0]) * np.sum((dx_pred - mean_dx) * (dy_pred - mean_dy))
            # test = np.abs(2 * (dx_pred / (dx_pred ** 2 + dy_pred ** 2)) * (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) * covariance_xy)
            # label_std_pred = np.sqrt(np.abs(( dx_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dy_std_pred ** 2 +
            #                          (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) ** 2 * dx_std_pred ** 2 +
            #                          2 * (dx_pred / (dx_pred ** 2 + dy_pred ** 2)) * (-dy_pred / (dx_pred ** 2 + dy_pred ** 2)) * covariance_xy))
            # label_std_pred = np.clip(label_std_pred, None, math.pi/float(2))

            prop_true, prop_pred, prop_pred_err = azimuth_true, azimuth_pred, azimuth_pred_err

        else:  # zenith
            zenith_true = convert_vectorial_to_spherical_dir(true, 'zenith', col_name_prefix='')
            zenith_pred = convert_vectorial_to_spherical_dir(pred, 'zenith', col_name_prefix='pred_')

            # clip std prediction to 0 (maybe not really necessary) and apply correction
            dz_pred_err = np.clip(np.copy(dz_pred_err), 0, None) * correction

            # error propagation with correlations neglected (covariance term = 0)
            zenith_pred_err = np.sqrt((-1 / np.sqrt((1 - dz_pred ** 2))) ** 2 * dz_pred_err ** 2)  # zen = arccos(z/norm(r))

            # # If we don't want to neglect the convariance term:
            # dx_std_pred, dy_std_pred = np.clip(dx_std_pred, 0, None) * correction, np.clip(dy_std_pred, 0, None) * correction
            # dx_pred, dy_pred = np.clip(dx_pred, -0.999, 0.999), np.clip(dy_pred, -0.999, 0.999)
            # label_std_pred = np.sqrt( (np.sqrt(dx_pred ** 2 + dy_pred ** 2) / (dx_pred ** 2 + dy_pred ** 2 + dz_pred ** 2))**2 * dz_std_pred ** 2 +
            #                           ((- dx_pred * dz_pred) / (np.sqrt(dx_pred ** 2 + dy_pred ** 2) * (dx_pred ** 2 + dy_pred ** 2 + dz_pred ** 2))) ** 2 * dx_std_pred ** 2 +
            #                           ((- dy_pred * dz_pred) / (np.sqrt(dx_pred ** 2 + dy_pred ** 2) * (dx_pred ** 2 + dy_pred ** 2 + dz_pred ** 2))) ** 2 * dy_std_pred ** 2)

            prop_true, prop_pred, prop_pred_err = zenith_true, zenith_pred, zenith_pred_err

    else:
        if prop_name == 'bjorkeny':
            # correct true by to 1 for e-NC events for the plotting
            abs_particle_type, is_cc = np.abs(mc_info['particle_type']), mc_info['is_cc']
            is_e_nc = np.logical_and(abs_particle_type == 12, is_cc == 0)

            prop_true = np.copy(true[properties[prop_name]['col_name_true']])  # TODO test if this really works or if this is actually necessary
            prop_true[is_e_nc] = 1
        else:
            prop_true = true[properties[prop_name]['col_name_true']]

        prop_pred = true[properties[prop_name]['col_name_pred']]
        prop_pred_err = true[properties[prop_name]['col_name_pred_err']]

        prop_pred_err = prop_pred_err * 1.253  # sqrt(pi/2) magic number correction with mse training
        prop_pred_err = np.abs(prop_pred_err)  # TODO necessary? -> rather set to zero, regarding the loss function?

    return prop_true, prop_pred, prop_pred_err


def plot_1d_reco_err_to_reco_residual_for_prop(prop_true, prop_pred, prop_pred_err, n_x_bins, fig, ax, title,
                                               pdf_plots, prop_name, unit):
    """
    Plots the true standard deviation of the error residuals (abs(true-pred)) versus
    the estimated uncertainty sigma_pred by a nn to a mpl ax and saves it to a mpl PdfPages instance.

    Parameters
    ----------
    prop_true : ndarray(ndim=1)
        Array which contains the true values of the specified property.
    prop_pred : ndarray(ndim=1)
        Array which contains the predicted values of the specified property.
    prop_pred_err : ndarray(ndim=1)
        Array which contains the predicted error values of the specified property, multiplied by the magic number.
    n_x_bins : int
        Number of x-bins.
    fig : mpl figure
        Matplotlib instance of a figure.
    ax : mpl.axes
        Axes object that refers to ax of an existing plt.sublots object.
    title : str
        Title that should be used for the plot.
    pdf_plots : mpl.backends.backend_pdf.PdfPages
        Instance of a matplotlib PdfPages object, to which the plot should be saved.
    prop_name : str
        Name of the property for that this plot should be made.
    unit : str
        String, which specifies the unit of the property and gets appended to the x and y axis labels.

    """
    prop_pred_std_range = (np.amin(prop_pred_err), np.amax(prop_pred_err))
    x_bins_std_pred = np.linspace(prop_pred_std_range[0], prop_pred_std_range[1], n_x_bins + 1)

    x, y = [], []
    for i in range(x_bins_std_pred.shape[0] - 1):  # same as n_x_bins
        prop_std_pred_low, prop_std_pred_high = x_bins_std_pred[i], x_bins_std_pred[i+1]
        prop_std_pred_mean = (prop_std_pred_low + prop_std_pred_high) / float(2)

        prop_std_pred_boolean_mask = np.logical_and(prop_std_pred_low < prop_pred_err, prop_pred_err <= prop_std_pred_high)
        if np.count_nonzero(prop_std_pred_boolean_mask) < 100:
            continue

        if prop_name == 'azimuth':
            # if abs(residual_azimuth) <= pi, take az_true - az_pred ; if residual_azimuth > pi take
            all_residuals = prop_true[prop_std_pred_boolean_mask] - prop_pred[prop_std_pred_boolean_mask]
            larger_pi = np.abs(all_residuals) > math.pi
            sign_larger_pi = np.sign(all_residuals[larger_pi])  # loophole for residual == 0, but doesn't matter below
            all_residuals[larger_pi] = sign_larger_pi * 2 * math.pi - (all_residuals[larger_pi])  # need same sign for 2pi compared to all_residuals value
            # print np.amin(all_residuals), np.amax(all_residuals)

            residuals_std = np.std(all_residuals)

        else:
            residuals_std = np.std(prop_true[prop_std_pred_boolean_mask] - prop_pred[prop_std_pred_boolean_mask])

        x.append(prop_std_pred_mean)
        y.append(residuals_std)

    plt.scatter(x, y, s=20, lw=0.75, c='blue', marker='+')

    title = plt.title(title)
    title.set_position([.5, 1.04])
    ax.set_xlabel(r'Estimated uncertainty $\sigma_{pred}$ ' + unit), ax.set_ylabel('Standard deviation of residuals ' + unit)
    ax.grid(True)

    pdf_plots.savefig(fig)
    ax.cla()


def make_2d_true_reco_plot_different_sigmas(pred_file, savefolder, savename, cuts=None):
    """
    Makes 2d true to reco plots of all properties in the properties dict, and selects only
    the best percentage of events based on the predicted errors by a nn.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    savename : str
        Savename with which the plots file should be saved.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.

    """
    fig, ax = plt.subplots()
    pdf_plots = mpl.backends.backend_pdf.PdfPages(savefolder + '/' + savename + '.pdf')

    ic_list = {'muon-CC': {'title': 'Track like (' + r'$\nu_{\mu}-CC$)'},
               'elec-CC': {'title': 'Shower like (' + r'$\nu_{e}-CC$)'},
               'elec-NC': {'title': 'Track like (' + r'$\nu_{e}-NC$)'},
               'tau-CC': {'title': 'Tau like (' + r'$\nu_{\tau}-CC$)'}}

    properties = {'energy': {'bins': np.linspace(1, 100, 100), 'axis_prop_info': ('energy', 'GeV')},
                  'azimuth': {'bins': np.linspace(-math.pi, math.pi, 100), 'axis_prop_info': ('azimuth', 'rad')},
                  'zenith': {'bins': np.linspace(-math.pi/float(2), math.pi/float(2), 100), 'axis_prop_info': ('azimuth', 'rad')}}

    mc_info, pred, true = pred_file['mc_info'], pred_file['pred'], pred_file['true']
    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        pred = pred[evt_sel_mask]
        true = true[evt_sel_mask]

    for prop_name in properties:
        for ic in ic_list:
            is_ic = select_ic(mc_info['particle_type'], mc_info['is_cc'], ic)
            if bool(np.any(is_ic, axis=0)) is False:
                continue

            mc_info_ic = mc_info[is_ic]
            pred_ic, true_ic = pred[is_ic], true[is_ic]
            prop_true, prop_pred, prop_pred_err = get_prop_true_pred_and_prop_pred_sigma(pred_ic, true_ic, mc_info_ic,
                                                                                         properties, prop_name)

            bins = properties[prop_name]['bins']
            title = ic_list[ic]['title']
            percentage_of_evts = [1.0, 0.8, 0.5, 0.2]

            plot_2d_dir_correlation_different_sigmas(prop_true, prop_pred, prop_pred_err, prop_name, properties,
                                                     percentage_of_evts, title, bins, fig, ax, pdf_plots)

    plt.close()
    pdf_plots.close()


def plot_2d_dir_correlation_different_sigmas(prop_true, prop_pred, prop_pred_err, prop_name, properties,
                                             percentage_of_evts, title, bins, fig, ax, pdf_plots):
    """
    Plots a 2d true to reco plot for a certain property, and selects only
    the best percentage of events based on the predicted errors by a nn.

    Parameters
    ----------
    prop_true : ndarray(ndim=1)
        Array which contains the true values of the specified property.
    prop_pred : ndarray(ndim=1)
        Array which contains the predicted values of the specified property.
    prop_pred_err : ndarray(ndim=1)
        Array which contains the predicted error values of the specified property, multiplied by the magic number.
    prop_name : str
        Name of the property for that this plot should be made.
    properties : dict
        Dict which contains some configurations of all properties.
    percentage_of_evts : list
        List, which contains the percentage of best events. For each value in the list, a separate plot is made.
    title : str
        Title that should be used for the plot.
    bins : ndarray(ndim=1)
        Bins that should be used for both the X and Y axis.
    fig : mpl figure
        Matplotlib instance of a figure.
    ax : mpl.axes
        Axes object that refers to ax of an existing plt.sublots object.
    pdf_plots : mpl.backends.backend_pdf.PdfPages
        Instance of a matplotlib PdfPages object, to which the plot should be saved.

    """
    cbar_max = None
    for i, percentage in enumerate(percentage_of_evts):

        n_total_evt = prop_pred_err.shape[0]
        n_events_to_keep = int(n_total_evt * percentage)  # how many events should be kept

        if prop_name == 'energy':  # TODO needs to be improved, maybe fit line through energyres per escale and divide by this func?
            std_pred_div_by_e_reco = np.divide(prop_pred_err, prop_pred)
            indices_events_to_keep = sorted(std_pred_div_by_e_reco.argsort()[:n_events_to_keep])  # select n minimum values in array

        else:
            indices_events_to_keep = sorted(prop_pred_err.argsort()[:n_events_to_keep])  # select n minimum values in array

        hist_2d_prop = np.histogram2d(prop_true[indices_events_to_keep], prop_pred[indices_events_to_keep], bins)
        bin_edges_prop = hist_2d_prop[1]

        if i == 0:
            cbar_max = hist_2d_prop[0].T.max()

        prop_true_to_reco = ax.pcolormesh(bin_edges_prop, bin_edges_prop, hist_2d_prop[0].T,
                                          norm=mpl.colors.LogNorm(vmin=1, vmax=cbar_max))

        plot_line_through_the_origin(prop_name, prop_name)

        title = plt.title('OrcaNet: ' + title + ', ' + str(int(percentage * 100)) + '% of total events')
        title.set_position([.5, 1.04])
        cbar = fig.colorbar(prop_true_to_reco, ax=ax)
        cbar.ax.set_ylabel('Number of events')

        axis_prop_info = properties[prop_name]['axis_prop_info']
        ax.set_xlabel('True ' + axis_prop_info[0] + ' [' + axis_prop_info[1] + ']')
        ax.set_ylabel('Reconstructed ' + axis_prop_info[0] + ' [' + axis_prop_info[1] + ']')
        plt.xlim(bins[0], bins[-1])
        plt.ylim(bins[0], bins[-1])

        # plt.tight_layout()
        pdf_plots.savefig(fig)

        if i > 0:
            # 1st plot, plot transparency of 100%
            hist_2d_prop_all = np.histogram2d(prop_true, prop_pred, bins)
            bin_edges_prop_all = hist_2d_prop_all[1]

            # only plot the bins that are not anymore in the histogram from above
            hist_2d_prop_larger_zero = np.invert(hist_2d_prop[0] == 0)
            hist_2d_prop_all[0][hist_2d_prop_larger_zero] = 0  # set everything to 0, when these bins are still > 0 in hist_2d_prop

            ax.pcolormesh(bin_edges_prop_all, bin_edges_prop_all, hist_2d_prop_all[0].T, alpha=0.5,
                          norm=mpl.colors.LogNorm(vmin=1, vmax=cbar_max))
            pdf_plots.savefig(fig)
            cbar.remove()
            plt.cla()

            # 2nd plot
            # only divide nonzero bins of hist_2d_prop_all
            hist_2d_prop_all = np.histogram2d(prop_true, prop_pred, bins)
            non_zero = hist_2d_prop_all[0] > 0
            hist_2d_all_div_leftover = np.copy(hist_2d_prop_all[0])
            hist_2d_all_div_leftover[non_zero] = np.divide(hist_2d_prop[0][non_zero], hist_2d_prop_all[0][non_zero])
            # hist_2d_all_div_leftover[np.invert(non_zero)] = -1
            corr_all_div_leftover = ax.pcolormesh(bin_edges_prop_all, bin_edges_prop_all, hist_2d_all_div_leftover.T, vmin=0, vmax=1)
            cbar_2 = fig.colorbar(corr_all_div_leftover, ax=ax)
            cbar_2.ax.set_ylabel('Fraction of leftover events')

            title = plt.title('OrcaNet: ' + title + ', ' + str(int(percentage * 100)) + '% of total events')
            title.set_position([.5, 1.04])
            ax.set_xlabel('True ' + axis_prop_info[0] + ' [' + axis_prop_info[1] + ']')
            ax.set_ylabel('Reconstructed ' + axis_prop_info[0] + ' [' + axis_prop_info[1] + ']')

            pdf_plots.savefig(fig)
            cbar_2.remove()
            plt.cla()

        if i == 0:
            cbar.remove()
        plt.cla()


# --------------------------- Code for error plots --------------------------- #


# --------------------------- Utility code --------------------------- #


def plot_line_through_the_origin(prop_1_name, prop_2_name):
    """
    Plots a line through the origin, if the two propertie names are recognized.

    Parameters
    ----------
    prop_1_name : str
        Name of the first property, e.g. 'energy_reco'.
    prop_2_name : str
        Name of the second property.

    """
    if 'azimuth' in prop_1_name and 'azimuth' in prop_2_name:
        pi = math.pi
        plt.plot([-pi, pi], [-pi, pi], 'k-', lw=1, zorder=10)

    elif 'zenith' in prop_1_name and 'zenith' in prop_2_name:
        pi = math.pi
        plt.plot([-pi/float(2), pi/float(2)], [-pi/float(2), pi/float(2)], 'k-', lw=1, zorder=10)

    elif 'bjorkeny' in prop_1_name and 'bjorkeny' in prop_2_name:
        plt.plot([0, 1], [0, 1], 'k-', lw=1, zorder=10)

    elif 'energy' in prop_1_name and 'energy' in prop_2_name:
        plt.plot([-1, 1], [-1, 1], 'k-', lw=1, zorder=10)

    else:
        warnings.warn('Couldnt decipher how to plot the line through the origin for your two properties'
                      ' that are to plot, ' + prop_1_name + ' & ' + prop_2_name + '.'
                      ' Plotting no line through the origin.')


def convert_vectorial_to_spherical_dir(vect_arr, spherical_coord_name, col_name_prefix=''):
    """
    Returns directions in azimuth / zenith based on an input array with vectorial (dir_x, dir_y, dir_z) coordinates.

    Parameters
    ----------
    vect_arr : ndarray(ndim=2)
        Structured array, which must contain dir_x, dir_y, dir_z
    spherical_coord_name : str
        String which specifies if azimuth or zenith should be returned.
    col_name_prefix : str
        String prefix that should be used for accessing the dir_x/dir_y/dir_z columns.

    Returns
    -------
    azimuth/zenith : ndarray(ndim=1)
        Array with azimuth / zenith spherical directions.

    """
    if spherical_coord_name == 'azimuth':
        # atan2(y,x)

        dir_x = vect_arr[col_name_prefix + 'dir_x']
        dir_y = vect_arr[col_name_prefix + 'dir_y']
        azimuth = np.arctan2(dir_y, dir_x)
        return azimuth

    elif spherical_coord_name == 'zenith':
        # atan2(z, sqrt(x**2 + y**2))

        dir_x = vect_arr[col_name_prefix + 'dir_x']
        dir_y = vect_arr[col_name_prefix + 'dir_y']
        dir_z = vect_arr[col_name_prefix + 'dir_z']

        zenith = np.arctan2(dir_z, np.sqrt(np.power(dir_x, 2) + np.power(dir_y, 2)))
        return zenith

    else:
        raise ValueError('Vectorial to spherical conversions other than azimuth and zenith are not known, you '
                         'specified ' + str(spherical_coord_name))


# --------------------------- Utility code --------------------------- #
