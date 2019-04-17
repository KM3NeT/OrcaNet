#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making plots for background classifiers (muon/random_noise/neutrinos).
"""

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from orcanet_contrib.plotting.utils import get_event_selection_mask, in_nd


def make_prob_hists_bg_classifier(pred_file, savefolder, bg_classes=('not_neutrino', 'neutrino'),
                                  cuts=None, savename_prefix=None, x_ranges=((0, 1),), xlabel_prefix='OrcaNet'):
    """
    Function that makes plots for the reco probability distributions of the background classifier.

    Makes one plot for each probability output (prob_muon, prob_random_noise, prob_neutrino)

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the background classification predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    bg_classes : tpl(str)
        Tuple that specifies the single classes of a background classifier.
        This is needed to find the arrays for the class probabilities in the pred dataset.
        I.e. if you supply bg_classes=('not_neutrino', 'neutrino'), the pred dataset should
        contain the rows "pred_not_neutrino" and "pred_neutrino".
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.
    savename_prefix : None/str
        Optional string prefix for the savename of the plot.
    x_ranges : list(tpl)
        List of tuples with various x_ranges that should be made of the same plot.
    xlabel_prefix : str
        Prefix that is in front of the x_axis label.

    """
    def configure_and_save_plot():
        """
        Configure and save a mpl plot with GridLines, Logscale etc. for the background classifier probability plots.
        """
        axes.legend(loc='upper center', ncol=1)
        plt.grid(True, zorder=0, linestyle='dotted')
        plt.yscale('log')

        plt.minorticks_on()

        plt.xlabel(xlabel_prefix + ' ' + bg_class + ' probability')
        plt.ylabel('Normed Quantity')
        title = plt.title('Probability to be classified as ' + bg_class)
        title.set_position([.5, 1.04])

        pdf_plots.savefig(fig)

    ptype, is_cc = pred_file['mc_info']['particle_type'], pred_file['mc_info']['is_cc']
    if cuts is not None:
        assert isinstance(cuts, str)
        print('Event number before selection: ' + str(ptype.shape))
        evt_sel_mask = get_event_selection_mask(pred_file['mc_info'], cut_name=cuts)
        ptype, is_cc = ptype[evt_sel_mask], is_cc[evt_sel_mask]
        print('Event number after selection: ' + str(ptype.shape))

    for bg_class in bg_classes:
        prob_class = pred_file['pred']['prob_' + bg_class]
        if cuts is not None:
            prob_class = prob_class[evt_sel_mask]

        savename = 'prob_' + bg_class if savename_prefix is None else savename_prefix + '_prob_' + bg_class
        pdf_plots = PdfPages(savefolder + '/' + savename + '.pdf')

        for x_range in x_ranges:
            fig, axes = plt.subplots()
            make_prob_hists_for_class(prob_class, ptype, is_cc, axes, x_range=x_range)
            configure_and_save_plot()
            plt.close()

        pdf_plots.close()


def select_class(class_name, ptype, is_cc):
    """
    Returns a boolean array which specifies, which rows in the ptype & is_cc 1d arrays belong to the class.

    Parameters
    ----------
    class_name : str
        String that specifies the class that should be selected for the boolean flag output.
    ptype : ndarray(ndim=1)
        Array with particle_types of some events.
    is_cc : ndarray(ndim=1)
        Array with is_cc of some events.

    Returns
    -------
    is_class : ndarray(ndim=1)
        Boolean flag, which specifies if each row belongs to the class specified by the class_name.

    """
    if class_name == 'mupage':
        is_class = np.abs(ptype) == 13
    elif class_name == 'random_noise':
        is_class = np.abs(ptype) == 0
    elif class_name == 'neutrino':
        is_class = np.logical_or.reduce((np.abs(ptype) == 12, np.abs(ptype) == 14, np.abs(ptype) == 16))
    elif class_name == 'neutrino_track':
        is_class = np.logical_and(np.abs(ptype) == 14, is_cc == 1)
    elif class_name == 'neutrino_shower':
        is_muon_nc = np.logical_and(np.abs(ptype) == 14, is_cc == 0)
        is_class = np.logical_or.reduce((np.abs(ptype) == 12, np.abs(ptype) == 16, is_muon_nc))
    else:
        raise ValueError('The class ' + str(class_name) + ' is not available.')

    return is_class


def make_prob_hists_for_class(prob_class, ptype, is_cc, axes, x_range=(0, 1)):
    """
    Makes a mpl hist plot of the probabilities for a certain probability class (either muon, random_noise or neutrino).

    Plots the probabilities for this prob_class for all events (muon, random_noise, neutrinos)

    Parameters
    ----------
    prob_class : ndarray(ndim=1)
        Reco probabilities for events for a certain probability class (muon, random_noise, neutrinos).
    ptype : ndarray(ndim=1)
        Array with particle_types of the same events as in the prob_class.
    is_cc : ndarray(ndim=1)
        Array with is_cc of the same events as in the prob_class.
    axes : mpl.axes
        Axes object that refers to ax of an existing plt.sublots object.
    x_range : tpl
        Sets the x_range of the prob hist plot.

    """
    is_mupage = select_class('mupage', ptype, is_cc)
    is_random_noise = select_class('random_noise', ptype, is_cc)
    is_neutrino = select_class('neutrino', ptype, is_cc)

    if x_range != (0, 1):
        x_range_diff = x_range[1] - x_range[0]
        bins = 100 * int(1 / x_range_diff)
    else:
        bins = 100

    axes.hist(prob_class[is_mupage], density=True, bins=bins, range=(0, 1), color='b', label='mupage', histtype='step', linestyle='-', zorder=3)
    axes.hist(prob_class[is_random_noise], density=True, bins=bins, range=(0, 1), color='r', label='random_noise', histtype='step', linestyle='-', zorder=3)
    axes.hist(prob_class[is_neutrino], density=True, bins=bins, range=(0, 1), color='saddlebrown', label='neutrino', histtype='step', linestyle='-', zorder=3)

    if x_range != (0, 1):
        axes.set_xlim(x_range)


def make_contamination_to_neutrino_efficiency_plot(pred_file, pred_file_std_reco, dataset_modifier, savefolder):
    """
    Makes bg classifier plots like muon contamination vs neutrino efficiency, both weighted 1 year and non weighted.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the background classification predictions of a nn model.
    pred_file_std_reco : h5py.File
        H5py file instance of a second reco result, typically the results of the standard reconstruction.
    dataset_modifier : str
        Specifies which dataset_modifier has been used in creating both pred files.
    savefolder : str
        Full dirpath of the folder where all the plots of this function should be saved.

    """
    mc_info, mc_info_std, pred, pred_std = select_events_that_exist_in_both_pred_files(pred_file, pred_file_std_reco)

    if dataset_modifier == 'bg_classifier_2_class':
        pdf_plots = PdfPages(savefolder + '/muon_contamination_to_neutr_efficiency.pdf')
        plot_contamination_to_neutr_eff_multi_e_cut(mc_info, mc_info_std, pred, pred_std, pdf_plots)

        pdf_plots.close()


def select_events_that_exist_in_both_pred_files(pred_file, pred_file_2):
    """
    Function that applies an event selection.

    Events are selected, if they are both found in the pred_file and pred_file_std_reco.
    Then, the selection is applied to BOTH pred_files of the input.

    TODO could be done without using the get_event_selection_mask!

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance of a prediction file.
    pred_file_2 : h5py.File
        H5py file instance of another prediction file.

    Returns
    -------
    mc_info : ndarray(ndim=2)
        Structured array containing the MC info of file 1 with the applied event selection.
    mc_info_2 : ndarray(ndim=2)
        Structured array containing the MC info of file 2 with the applied event selection.
    pred : ndarray(ndim=2)
        Structured array containing the pred dataset of file 1 with the applied event selection.
    pred_2 : ndarray(ndim=2)
        Structured array containing the pred dataset of file 2 with the applied event selection.

    """
    # select pred_file evts that are in the std reco summary files
    print('Shape of mc_info pred_file_1 before selection: ' + str(pred_file['mc_info'].shape))
    evt_sel_mask_dl_in_std = get_event_selection_mask(pred_file['mc_info'], cut_name='bg_classifier')

    mc_info = pred_file['mc_info'][evt_sel_mask_dl_in_std]
    pred = pred_file['pred'][evt_sel_mask_dl_in_std]
    print('Shape of mc_info pred_file_1 after selection: ' + str(mc_info.shape))

    # get rid of duplicates!
    print('Shape of mc_info pred_file_1 before deselecting the few duplicates: ' + str(mc_info.shape))
    unq, idx, count = np.unique(mc_info[['run_id', 'event_id', 'prod_ident', 'particle_type', 'is_cc']], axis=0, return_counts=True, return_index=True)
    # select only unique rows
    mask = np.zeros(mc_info.shape[0], np.bool)
    mask[idx] = 1
    mc_info = mc_info[mask]
    print('Shape of mc_info pred_file_1 after deselecting the few duplicates: ' + str(mc_info.shape))
    pred = pred[mask]

    # select pred_file_2 events that are in the already cut pred_file
    ax = np.newaxis
    mc_info_2 = pred_file_2['mc_info']

    mc_info_dl_reco = np.concatenate([mc_info['run_id'][:, ax], mc_info['event_id'][:, ax],
                                      mc_info['prod_ident'][:, ax], mc_info['particle_type'][:, ax],
                                      mc_info['is_cc'][:, ax]], axis=1)

    mc_info_std_reco_selection = np.concatenate([mc_info_2['run_id'][:, ax], mc_info_2['event_id'][:, ax],
                                                 mc_info_2['prod_ident'][:, ax], mc_info_2['particle_type'][:, ax],
                                                 mc_info_2['is_cc'][:, ax]], axis=1)

    print('Shape of mc_info pred_file_2 (std reco) before selection: ' + str(pred_file_2['mc_info'].shape))
    evt_sel_mask_std_in_dl = in_nd(mc_info_std_reco_selection, mc_info_dl_reco)
    mc_info_2 = pred_file_2['mc_info'][evt_sel_mask_std_in_dl]
    print('Shape of mc_info pred_file_2 (std reco) after selection: ' + str(mc_info_2.shape))
    pred_2 = pred_file_2['pred'][evt_sel_mask_std_in_dl]

    return mc_info, mc_info_2, pred, pred_2


def plot_contamination_to_neutr_eff_multi_e_cut(mc_info, mc_info_std, pred, pred_std, pdf_plots):
    """
    Make muon/random-noise contamination plots vs neutrino efficiency.

    TODO improve readability of function.

    Parameters
    ----------
    mc_info : ndarray(ndim=2)
        Structured array containing the MC info of reco 1.
    mc_info_std : ndarray(ndim=2)
        Structured array containing the MC info of reco 2 (typically standard reco).
    pred : ndarray(ndim=2)
        Structured array containing the pred info of reco 1.
    pred_std : ndarray(ndim=2)
        Structured array containing the pred info of reco 2 (typically standard reco).
    pdf_plots : mpl.backends.backend_pdf.PdfPages
        Matplotlib pdfpages instance, to which the plots should be saved.

    """
    def make_plots():
        fig, ax = plt.subplots()
        # -- Make non weighted plots -- #
        ax.plot(np.array(x_contamination_mupage), np.array(y_neutrino_efficiency), label='OrcaNet')
        ax.plot(np.array(x_contamination_mupage_std), np.array(y_neutrino_efficiency_std), label='Std Reco')

        ax.set_xlim(left=0, right=0.02)
        ax.set_ylim(bottom=80, top=100.5)
        ax.set_xlabel('Fraction of muons in dataset [%]'), ax.set_ylabel('Percentage of surviving neutrinos [%]')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.title('E-range:' + str(tpl) + ', N_muon: ' + str(n_muons_total) + ', N_neutrinos: ' + str(n_neutrinos_total_e_cut))

        pdf_plots.savefig(fig)
        ax.cla()

        #  make neutr eff vs. n_muons_left plot
        ax.plot(np.array(y_neutrino_efficiency), np.array(y_n_muons), label='OrcaNet')
        ax.plot(np.array(y_neutrino_efficiency_std), np.array(y_n_muons_std), label='Std Reco')

        ax.set_yscale('log')
        ax.set_xlabel('Percentage of surviving neutrinos [%]'), ax.set_ylabel('Number of muons left (total: ' + str(n_muons_total) + ')')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.title('E-range:' + str(tpl) + ', N_muon: ' + str(n_muons_total) + ', N_neutrinos: ' + str(n_neutrinos_total_e_cut))

        #ax.set_xlim(left=80, right=100)

        pdf_plots.savefig(fig)
        plt.close()

        # -- Make weighted plots -- #
        # Muon contamination
        fig, ax = plt.subplots()

        ax.plot(np.array(ax_muon_contamination_weighted), np.array(ax_neutrino_efficiency_weighted), label='OrcaNet')
        ax.plot(np.array(ax_muon_contamination_weighted_std), np.array(ax_neutrino_efficiency_weighted_std), label='Std Reco')

        ax.set_xlim(left=0, right=20)
        ax.set_xlabel('Muon contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.title('E-range:' + str(tpl) + ', weighted for one year')

        pdf_plots.savefig(fig)

        # Random noise contamination
        fig, ax = plt.subplots()

        ax.plot(np.array(ax_rn_contamination_weighted), np.array(ax_neutrino_efficiency_weighted), label='OrcaNet')
        ax.plot(np.array(ax_rn_contamination_weighted_std), np.array(ax_neutrino_efficiency_weighted_std), label='Std Reco')

        ax.set_xlim(left=0, right=5)
        ax.set_xlabel('Random noise contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.title('E-range:' + str(tpl) + ', weighted for one year')

        pdf_plots.savefig(fig)
        plt.close()

    ptype_dl, is_cc_dl = mc_info['particle_type'], mc_info['is_cc']
    ptype_std, is_cc_std = mc_info_std['particle_type'], mc_info_std['is_cc']
    is_neutrino, is_neutrino_std = select_class('neutrino', ptype_dl, is_cc_dl), select_class('neutrino', ptype_std, is_cc_std)
    is_mupage, is_mupage_std = select_class('mupage', ptype_dl, is_cc_dl), select_class('mupage', ptype_std, is_cc_std)
    is_rn, is_rn_std = select_class('random_noise', ptype_dl, is_cc_dl), select_class('random_noise', ptype_std, is_cc_std)

    n_neutrinos_total = np.count_nonzero(is_neutrino)  # dl or std doesnt matter for n_neutrinos_total
    n_muons_total = np.count_nonzero(is_mupage)
    n_rn_total = np.count_nonzero(is_rn)
    print('Total number of neutrinos: ' + str(n_neutrinos_total) + ', ' + str(np.count_nonzero(is_neutrino_std)))
    print('Total number of mupage muons: ' + str(n_muons_total) + ', ' + str(np.count_nonzero(is_mupage_std)))
    print('Total number of random_noise: ' + str(n_rn_total) + ', ' + str(np.count_nonzero(is_rn_std)))

    # define e_cut and cut values on neutrino prob
    e_cuts = ((1, 100), (1, 5), (5, 10), (10, 20), (20, 100))
    cuts = np.logspace(-5.9, 1, num=20000)  # cuts = np.linspace(0, 1, 10000)
    prob_not_neutrino = pred['prob_not_neutrino']
    muon_score_std = pred_std['prob_muon']
    rn_noise_score_std = pred_std['prob_noise']
    w_1_y = mc_info['oscillated_weight_one_year_bg_sel']
    w_1_y_std = mc_info_std['oscillated_weight_one_year_bg_sel']

    for tpl in e_cuts:
        print('Making contamination plots for e-cut ' + str(tpl))
        e_low, e_high = tpl[0], tpl[1]
        e_cut_mask = np.logical_and(mc_info['energy'] >= e_low, mc_info['energy'] < e_high)
        e_cut_mask_std = np.logical_and(mc_info_std['energy'] >= e_low, mc_info_std['energy'] < e_high)

        # make energy cuts on neutrinos only
        is_neutrino_e_cut = np.logical_and(e_cut_mask, is_neutrino)
        is_neutrino_e_cut_std = np.logical_and(e_cut_mask_std, is_neutrino_std)

        n_neutrinos_total_e_cut = np.count_nonzero(is_neutrino_e_cut)  # should be same for both dl and std
        n_neutrinos_total_e_cut_weighted = np.sum(w_1_y[is_neutrino_e_cut])

        n_neutrinos_total_e_cut_weighted_std = np.sum(w_1_y_std[is_neutrino_e_cut_std])
        assert np.round(n_neutrinos_total_e_cut_weighted_std, 5) == np.round(n_neutrinos_total_e_cut_weighted, 5)

        x_contamination_mupage, y_neutrino_efficiency, y_n_muons = [], [], []
        x_contamination_mupage_std, y_neutrino_efficiency_std, y_n_muons_std = [], [], []

        ax_neutrino_efficiency_weighted, ax_muon_contamination_weighted = [], []
        ax_neutrino_efficiency_weighted_std, ax_muon_contamination_weighted_std = [], []
        ax_rn_contamination_weighted, ax_rn_contamination_weighted_std = [], []

        for i in range(len(cuts)):
            # -- non weighted -- #
            # dl, how many events are left for the first (dl) pred array?
            n_neutrino = np.count_nonzero(prob_not_neutrino[is_neutrino_e_cut] < cuts[i])
            n_muons = np.count_nonzero(prob_not_neutrino[is_mupage] < cuts[i])

            if n_neutrino != 0:
                x_contamination_mupage.append((n_muons / n_neutrino) * 100)
                y_neutrino_efficiency.append((n_neutrino / n_neutrinos_total_e_cut) * 100)
                y_n_muons.append(n_muons)

            # std, how many events are left for std reco?
            n_neutrino_std = np.count_nonzero(muon_score_std[is_neutrino_e_cut_std] < cuts[i])
            n_muons_std = np.count_nonzero(muon_score_std[is_mupage_std] < cuts[i])

            if n_neutrino_std != 0:
                x_contamination_mupage_std.append((n_muons_std / n_neutrino_std) * 100)
                y_neutrino_efficiency_std.append((n_neutrino_std / n_neutrinos_total_e_cut) * 100)
                y_n_muons_std.append(n_muons_std)

            # -- weighted -- #
            # dl
            neutr_sel, mupage_sel, rn_sel = prob_not_neutrino[is_neutrino_e_cut] < cuts[i], prob_not_neutrino[is_mupage] < cuts[i], prob_not_neutrino[is_rn] < cuts[i]
            n_neutrino_weighted = np.sum(w_1_y[is_neutrino_e_cut][neutr_sel])
            n_mupage_weighted = np.sum(w_1_y[is_mupage][mupage_sel])
            n_rn_weighted = np.sum(w_1_y[is_rn][rn_sel])

            if n_neutrino_weighted != 0:
                ax_neutrino_efficiency_weighted.append((n_neutrino_weighted / n_neutrinos_total_e_cut_weighted) * 100)
                ax_muon_contamination_weighted.append((n_mupage_weighted / n_neutrino_weighted) * 100)
                ax_rn_contamination_weighted.append((n_rn_weighted / n_neutrino_weighted) * 100)

            # std
            neutr_sel_std, mupage_sel_std, rn_sel_std = muon_score_std[is_neutrino_e_cut_std] < cuts[i], muon_score_std[is_mupage_std] < cuts[i], rn_noise_score_std[is_rn_std] < cuts[i]
            n_neutrino_weighted_std = np.sum(w_1_y_std[is_neutrino_e_cut_std][neutr_sel_std])
            n_mupage_weighted_std = np.sum(w_1_y_std[is_mupage_std][mupage_sel_std])
            n_rn_weighted_std = np.sum(w_1_y_std[is_rn_std][rn_sel_std])

            if n_neutrino_weighted_std != 0:
                ax_neutrino_efficiency_weighted_std.append((n_neutrino_weighted_std / n_neutrinos_total_e_cut_weighted) * 100)
                ax_muon_contamination_weighted_std.append((n_mupage_weighted_std / n_neutrino_weighted_std) * 100)
                ax_rn_contamination_weighted_std.append((n_rn_weighted_std / n_neutrino_weighted_std) * 100)

        # remove all 0 entries with n_muons, only needed for dl, since standard reco cant go to 0 muons anyways
        idx_last_zero = np.amax(np.where(np.array(y_n_muons) == 0))
        y_n_muons = y_n_muons[idx_last_zero:]
        y_neutrino_efficiency = y_neutrino_efficiency[idx_last_zero:]
        x_contamination_mupage = x_contamination_mupage[idx_last_zero:]

        make_plots()
