#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making plots for background classifiers (muon/random_noise/neutrinos).
"""

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from orcanet_contrib.plotting.utils import get_event_selection_mask, in_nd


def make_prob_hists_bg_classifier(pred_file, savefolder, bg_classes = ['not_neutrino', 'neutrino'], cuts=None, savename_prefix=None):
    """
    Function that makes plots for the reco probability distributions of the background classifier.

    Makes one plot for each probability output (prob_muon, prob_random_noise, prob_neutrino)

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the background classification predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.
    savename_prefix : None/str
        Optional string prefix for the savename of the plot.

    """
    def configure_and_save_plot(bg_class, savefolder, savename_prefix):
        """
        Configure and save a mpl plot with GridLines, Logscale etc. for the background classifier probability plots.
        Parameters
        ----------
        bg_class : str
            A string which specifies the probability class: (muon, random_noise, neutrino).
        savefolder : str
            Path to the directory, where the plot should be saved.
        savename_prefix : None/str
            Optional string prefix for the savename of the plot.

        """
        axes.legend(loc='upper center', ncol=1)
        plt.grid(True, zorder=0, linestyle='dotted')
        plt.yscale('log')

        x_ticks_major = np.arange(0, 1.1, 0.1)
        plt.xticks(x_ticks_major)
        plt.minorticks_on()

        plt.xlabel('OrcaNet ' + bg_class + ' probability')
        plt.ylabel('Normed Quantity')
        title = plt.title('Probability to be classified as ' + bg_class)
        title.set_position([.5, 1.04])

        savename = 'prob_' + bg_class if savename_prefix is None else savename_prefix + '_prob_' + bg_class
        plt.savefig(savefolder + '/' + savename + '.pdf')
        plt.savefig(savefolder + '/' + savename + '.png', dpi=600)

    fig, axes = plt.subplots()

    ptype, is_cc = pred_file['mc_info']['particle_type'], pred_file['mc_info']['is_cc']
    if cuts is not None:
        assert isinstance(cuts, str)
        print('Event number before selection: ' + str(ptype.shape))
        evt_sel_mask = get_event_selection_mask(pred_file['mc_info'], cut_name=cuts)
        ptype, is_cc = ptype[evt_sel_mask], is_cc[evt_sel_mask]
        print('Event number after selection: ' + str(ptype.shape))

    # bg_classes = ['muon', 'random_noise', 'neutrino']
    bg_classes = ['not_neutrino', 'neutrino']
    for bg_class in bg_classes:
        prob_class = pred_file['pred']['prob_' + bg_class]
        if cuts is not None:
            prob_class = prob_class[evt_sel_mask]

        make_prob_hists_for_class(prob_class, ptype, is_cc, axes, range=(0, 1))
        make_prob_hists_for_class(prob_class, ptype, is_cc, axes, range=(0.95, 1))
        configure_and_save_plot(bg_class, savefolder, savename_prefix)
        plt.cla()


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
        is_class = np.abs(ptype == 0)
    elif class_name == 'neutrino':
        is_class = np.logical_or.reduce((np.abs(ptype == 12), np.abs(ptype == 14), np.abs(ptype == 16)))
    elif class_name == 'neutrino_track':
        is_class = np.logical_and(np.abs(ptype == 14), is_cc == 1)
    elif class_name == 'neutrino_shower':
        is_muon_nc = np.logical_and(np.abs(ptype == 14), is_cc == 0)
        is_class = np.logical_or.reduce((np.abs(ptype == 12), np.abs(ptype == 16), is_muon_nc))
    else:
        raise ValueError('The class ' + str(class_name) + ' is not available.')

    return is_class


def make_prob_hists_for_class(prob_class, ptype, is_cc, axes, range=(0, 1)):
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

    """
    is_mupage = select_class('mupage', ptype, is_cc)
    is_random_noise = select_class('random_noise', ptype, is_cc)
    is_neutrino = select_class('neutrino', ptype, is_cc)

    axes.hist(prob_class[is_mupage], density=True, bins=100, range=range, color='b', label='mupage', histtype='step', linestyle='-', zorder=3)
    axes.hist(prob_class[is_random_noise], density=True, bins=100, range=range, color='r', label='random_noise', histtype='step', linestyle='-', zorder=3)
    axes.hist(prob_class[is_neutrino], density=True, bins=100, range=range, color='saddlebrown', label='neutrino', histtype='step', linestyle='-', zorder=3)


def make_contamination_to_neutrino_efficiency_plot(pred_file, pred_file_std_reco, dataset_modifier, savefolder):
    """

    Parameters
    ----------
    pred_file
    pred_file_std_reco
    dataset_modifier
    savefolder

    Returns
    -------

    """

    # select pred_file evts that are in the std reco summary files
    evt_sel_mask_dl_in_std = get_event_selection_mask(pred_file['mc_info'], cut_name='bg_classifier')
    mc_info = pred_file['mc_info'][evt_sel_mask_dl_in_std]
    pred = pred_file['pred'][evt_sel_mask_dl_in_std]

    # select pred_file_std_reco events that are in the already cut pred_file
    mc_info_std = pred_file_std_reco['mc_info']
    ax = np.newaxis

    mc_info_dl_reco = np.concatenate([mc_info['run_id'][:, ax], mc_info['event_id'][:, ax],
                                      mc_info['prod_ident'][:, ax], mc_info['particle_type'][:, ax],
                                      mc_info['is_cc'][:, ax]], axis=1)

    mc_info_std_reco_selection = np.concatenate([mc_info_std['run_id'][:, ax], mc_info_std['event_id'][:, ax],
                                                 mc_info_std['prod_ident'][:, ax], mc_info_std['particle_type'][:, ax],
                                                 mc_info_std['is_cc'][:, ax]], axis=1)

    evt_sel_mask_std_in_dl = in_nd(mc_info_std_reco_selection, mc_info_dl_reco)
    mc_info_std = pred_file_std_reco['mc_info'][evt_sel_mask_std_in_dl]
    pred_std = pred_file_std_reco['pred'][evt_sel_mask_std_in_dl]

    print(mc_info.shape)
    print(mc_info_std.shape)
    u, c = np.unique(mc_info_std, return_counts=True)
    dup = u[c > 1]
    np.save('/home/woody/capn/mppi033h/duplicates.npy', dup)
    print(dup)
    # assert np.count_nonzero(mc_info_std) == mc_info.shape[0]  # dl and std should have same number of events now TODO actually not, but only 100 of 400k

    # get plot data
    ptype_dl, is_cc_dl = mc_info['particle_type'], mc_info['is_cc']
    ptype_std, is_cc_std = mc_info_std['particle_type'], mc_info_std['is_cc']

    if dataset_modifier == 'bg_classifier_2_class':
        # plot some info about the dataset
        is_neutrino, is_neutrino_std = select_class('neutrino', ptype_dl, is_cc_dl), select_class('neutrino', ptype_std, is_cc_std)
        is_mupage, is_mupage_std = select_class('mupage', ptype_dl, is_cc_dl), select_class('mupage', ptype_std, is_cc_std)

        n_neutrinos_total = np.count_nonzero(is_neutrino)  # dl or std doesnt matter for n_neutrinos_total
        n_muons_total = np.count_nonzero(is_mupage)

        print('Total number of neutrinos: ' + str(n_neutrinos_total))
        print('Total number of mupage muons: ' + str(n_muons_total))

        pdf_plots = PdfPages(savefolder + '/muon_contamination_to_neutr_efficiency.pdf')
        plot_contamination_to_neutr_eff_multi_e_cut(mc_info, mc_info_std, pred, pred_std, is_neutrino, is_neutrino_std,
                                                    is_mupage, is_mupage_std, n_muons_total, pdf_plots)

        plt.close()
        pdf_plots.close()


def plot_contamination_to_neutr_eff_multi_e_cut(mc_info, mc_info_std, pred, pred_std, is_neutrino, is_neutrino_std,
                                                is_mupage, is_mupage_std, n_muons_total, pdf_plots):
    """

    Parameters
    ----------
    mc_info
    mc_info_std
    pred
    pred_std
    is_neutrino
    is_neutrino_std
    is_mupage
    is_mupage_std
    n_muons_total
    pdf_plots

    Returns
    -------

    """
    e_cuts = ((1, 100), (1, 5), (5, 10), (10, 20), (20, 100))
    cuts = np.linspace(0, 1, 5000)
    prob_not_neutrino = pred['prob_not_neutrino']
    muon_score_std = pred_std['prob_muon']

    for tpl in e_cuts:
        e_low, e_high = tpl[0], tpl[1]
        e_cut_mask = np.logical_and(mc_info['energy'] >= e_low, mc_info['energy'] < e_high)
        e_cut_mask_std = np.logical_and(mc_info_std['energy'] >= e_low, mc_info_std['energy'] < e_high)

        # make energy cuts on neutrinos only
        is_neutrino_e_cut = np.logical_and(e_cut_mask, is_neutrino)
        is_neutrino_e_cut_std = np.logical_and(e_cut_mask_std, is_neutrino_std)

        n_neutrinos_total_e_cut = np.count_nonzero(is_neutrino_e_cut)  # should be same for both dl and std

        x_contamination_mupage, y_neutrino_efficiency = [], []
        x_contamination_mupage_std, y_neutrino_efficiency_std = [], []
        for i in range(len(cuts)):
            # how many events are left for the first (dl) pred array?
            n_neutrino = np.count_nonzero(prob_not_neutrino[is_neutrino_e_cut] < cuts[i])
            n_muons = np.count_nonzero(prob_not_neutrino[is_mupage] < cuts[i])

            if n_muons != 0 and n_neutrino != 0:
                x_contamination_mupage.append((n_muons / n_neutrino) * 100)
                y_neutrino_efficiency.append((n_neutrino / n_neutrinos_total_e_cut) * 100)

            # how many events are left for std reco?
            n_neutrino_std = np.count_nonzero(muon_score_std[is_neutrino_e_cut_std] < cuts[i])
            n_muons_std = np.count_nonzero(muon_score_std[is_mupage_std] < cuts[i])

            if n_muons_std != 0 and n_neutrino_std != 0:
                x_contamination_mupage_std.append((n_muons_std / n_neutrino_std) * 100)
                y_neutrino_efficiency_std.append((n_neutrino_std / n_neutrinos_total_e_cut) * 100)

        fig, ax = plt.subplots()
        ax.plot(np.array(x_contamination_mupage), np.array(y_neutrino_efficiency), label='OrcaNet')
        ax.plot(np.array(x_contamination_mupage_std), np.array(y_neutrino_efficiency_std), label='Std Reco')

        # ax.set_xscale('log')
        ax.set_xlim(left=0, right=0.02)
        ax.set_ylim(bottom=90, top=100.5)
        ax.set_xlabel('Fraction of muons in dataset [%]'), ax.set_ylabel('Percentage of surviving neutrinos [%]')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.title('E-range:' + str(tpl) + ', N_muon: ' + str(n_muons_total) + ', N_neutrinos: ' + str(n_neutrinos_total_e_cut))

        pdf_plots.savefig(fig)
        ax.cla()
