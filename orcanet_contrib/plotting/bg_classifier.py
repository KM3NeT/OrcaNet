#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making plots for background classifiers (muon/random_noise/neutrinos).
"""

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from orcanet_contrib.plotting.utils import get_mc_info_and_other_datasets


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

    dsets = get_mc_info_and_other_datasets(pred_file, 'mc_info', 'pred', cuts=cuts)
    mc_info, pred = dsets[0], dsets[1]
    ptype, is_cc = mc_info['particle_type'], mc_info['is_cc']

    for bg_class in bg_classes:
        prob_class = pred['prob_' + bg_class]

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


def make_contamination_to_neutrino_efficiency_plot(pred_file_1, pred_file_2, dataset_modifier, savefolder, cuts=None):
    """
    Makes bg classifier plots like muon contamination vs neutrino efficiency, both weighted 1 year and non weighted.

    Parameters
    ----------
    pred_file_1 : h5py.File
        H5py file instance, which stores the background classification predictions of a nn model.
    pred_file_2 : h5py.File
        H5py file instance of a second reco result, typically the results of the standard reconstruction.
    dataset_modifier : str
        Specifies which dataset_modifier has been used in creating both pred files.
    savefolder : str
        Full dirpath of the folder where all the plots of this function should be saved.

    """
    mc_info_1, mc_info_2 = pred_file_1['mc_info'], pred_file_2['mc_info']
    pred_1, pred_2 = pred_file_1['pred'], pred_file_2['pred']

    if cuts is not None:
        assert type(cuts) is tuple
        mc_info_1, mc_info_2 = mc_info_1[cuts[0]], mc_info_2[cuts[1]]
        pred_1, pred_2 = pred_1[cuts[0]], pred_2[cuts[1]]

    if dataset_modifier == 'bg_classifier_2_class':
        pdf_plots = PdfPages(savefolder + '/muon_contamination_to_neutr_efficiency.pdf')
        plot_contamination_to_neutr_eff_multi_e_cut(mc_info_1, mc_info_2, pred_1, pred_2, pdf_plots)

        pdf_plots.close()


def plot_contamination_to_neutr_eff_multi_e_cut(mc_info_dl, mc_info_std, pred_dl, pred_std, pdf_plots,
                                                overlay=('KM3NeT Preliminary', (0.2, 0.7))):
    """
    Make muon/random-noise contamination plots vs neutrino efficiency.

    TODO improve readability of function.

    Parameters
    ----------
    mc_info_dl : ndarray(ndim=2)
        Structured array containing the MC info of reco 1.
    mc_info_std : ndarray(ndim=2)
        Structured array containing the MC info of reco 2 (typically standard reco).
    pred_dl : ndarray(ndim=2)
        Structured array containing the pred_dl info of reco 1.
    pred_std : ndarray(ndim=2)
        Structured array containing the pred_dl info of reco 2 (typically standard reco).
    pdf_plots : mpl.backends.backend_pdf.PdfPages
        Matplotlib pdfpages instance, to which the plots should be saved.

    """
    def make_plots():
        # -- Make weighted plots -- #

        # - Make continous plots
        # Muon contamination

        fig, ax = plt.subplots()

        ax.plot(np.array(muon_cont_weighted_dl), np.array(neutrino_eff_weighted_dl), label='CNN')
        ax.plot(np.array(muon_cont_weighted_std), np.array(neutrino_eff_weighted_std), label='RF')

        ax.set_xlim(left=0, right=20)
        ax.set_xlabel('Muon contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
        ax.grid(True)
        ax.legend(loc='center right')
        plt.title('E-range:' + str(tpl) + ', weighted for an atmospheric flux')

        pdf_plots.savefig(fig)
        plt.close()

        # Random noise contamination
        fig, ax = plt.subplots()

        ax.plot(np.array(rn_cont_weighted_dl), np.array(neutrino_eff_weighted_dl), label='CNN')
        ax.plot(np.array(rn_cont_weighted_std), np.array(neutrino_eff_weighted_std), label='RF')

        ax.set_xlim(left=0, right=5)
        ax.set_xlabel('Random noise contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
        ax.grid(True)
        ax.legend(loc='center right')
        plt.title('E-range:' + str(tpl) + ', weighted for 10kHz noise')

        pdf_plots.savefig(fig)
        plt.close()

        # - Make errorbar plots
        # Muon contamination

        fig, ax = plt.subplots()

        data_points_range_mupage = np.linspace(0, 20, 21)
        idx_mupage_closest_dl, idx_mupage_closest_std = [], []
        for j in range(len(data_points_range_mupage)):
            dpoint = data_points_range_mupage[j]

            idx_mupage_closest_dl.append((np.abs(np.array(muon_cont_weighted_dl) - dpoint)).argmin())
            idx_mupage_closest_std.append((np.abs(np.array(muon_cont_weighted_std) - dpoint)).argmin())

        dpoints_neutr_eff_dl = np.array(neutrino_eff_weighted_dl)[idx_mupage_closest_dl]
        dpoints_muon_cont_dl = np.array(muon_cont_weighted_dl)[idx_mupage_closest_dl]
        dpoints_muon_cont_err_dl = np.array(muon_cont_weighted_dl_err)[idx_mupage_closest_dl]

        dpoints_neutr_eff_std = np.array(neutrino_eff_weighted_std)[idx_mupage_closest_std]
        dpoints_muon_cont_std = np.array(muon_cont_weighted_std)[idx_mupage_closest_std]
        dpoints_muon_cont_err_std = np.array(muon_cont_weighted_std_err)[idx_mupage_closest_std]

        plt.errorbar(dpoints_muon_cont_dl, dpoints_neutr_eff_dl, xerr=dpoints_muon_cont_err_dl, fmt='.', markersize=4, label='CNN', linewidth=0.7)
        plt.errorbar(dpoints_muon_cont_std, dpoints_neutr_eff_std, xerr=dpoints_muon_cont_err_std, fmt='.', markersize=4, label='RF', linewidth=0.7)

        ax.set_xlim(left=0, right=20)
        ax.set_xlabel('Muon contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
        ax.grid(True)
        ax.legend(loc='center right')
        plt.title('E-range:' + str(tpl) + ', weighted for an atmospheric flux')
        plt.text(overlay[1][0], overlay[1][1], overlay[0], transform=ax.transAxes, weight='bold')

        pdf_plots.savefig(fig)

        # Just because we want a plot from 85 to 105
        ax.set_ylim(bottom=85, top=ax.get_ylim()[1])
        pdf_plots.savefig(fig)

        plt.close()

        # Random noise contamination

        fig, ax = plt.subplots()

        data_points_range_rn = np.linspace(0, 10, 21)
        idx_rn_closest_dl, idx_rn_closest_std = [], []
        for j in range(len(data_points_range_rn)):
            dpoint = data_points_range_rn[j]

            idx_rn_closest_dl.append((np.abs(np.array(rn_cont_weighted_dl) - dpoint)).argmin())
            idx_rn_closest_std.append((np.abs(np.array(rn_cont_weighted_std) - dpoint)).argmin())

        dpoints_neutr_eff_dl = np.array(neutrino_eff_weighted_dl)[idx_rn_closest_dl]
        dpoints_rn_cont_dl = np.array(rn_cont_weighted_dl)[idx_rn_closest_dl]
        dpoints_rn_cont_err_dl = np.array(rn_cont_weighted_dl_err)[idx_rn_closest_dl]

        dpoints_neutr_eff_std = np.array(neutrino_eff_weighted_std)[idx_rn_closest_std]
        dpoints_rn_cont_std = np.array(rn_cont_weighted_std)[idx_rn_closest_std]
        dpoints_rn_cont_err_std = np.array(rn_cont_weighted_std_err)[idx_rn_closest_std]

        plt.errorbar(dpoints_rn_cont_dl, dpoints_neutr_eff_dl, xerr=dpoints_rn_cont_err_dl, fmt='.', markersize=4, label='CNN', linewidth=0.7)
        plt.errorbar(dpoints_rn_cont_std, dpoints_neutr_eff_std, xerr=dpoints_rn_cont_err_std, fmt='.', markersize=4, label='RF', linewidth=0.7)

        ax.set_xlim(left=0, right=5)
        ax.set_xlabel('Random noise contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
        ax.grid(True)
        ax.legend(loc='center right')
        plt.title('E-range:' + str(tpl) + ', weighted for 10kHz noise')
        plt.text(overlay[1][0], overlay[1][1], overlay[0], transform=ax.transAxes, weight='bold')

        pdf_plots.savefig(fig)
        plt.close()

    ptype_dl, is_cc_dl = mc_info_dl['particle_type'], mc_info_dl['is_cc']
    ptype_std, is_cc_std = mc_info_std['particle_type'], mc_info_std['is_cc']
    is_neutrino_dl, is_neutrino_std = select_class('neutrino', ptype_dl, is_cc_dl), select_class('neutrino', ptype_std, is_cc_std)
    is_mupage_dl, is_mupage_std = select_class('mupage', ptype_dl, is_cc_dl), select_class('mupage', ptype_std, is_cc_std)
    is_rn_dl, is_rn_std = select_class('random_noise', ptype_dl, is_cc_dl), select_class('random_noise', ptype_std, is_cc_std)

    n_neutrinos_total = np.count_nonzero(is_neutrino_dl)  # dl or std doesnt matter for n_neutrinos_total
    n_muons_total = np.count_nonzero(is_mupage_dl)
    n_rn_total = np.count_nonzero(is_rn_dl)
    print('Total number of neutrinos: ' + str(n_neutrinos_total) + ', ' + str(np.count_nonzero(is_neutrino_std)))
    print('Total number of mupage muons: ' + str(n_muons_total) + ', ' + str(np.count_nonzero(is_mupage_std)))
    print('Total number of random_noise: ' + str(n_rn_total) + ', ' + str(np.count_nonzero(is_rn_std)))

    # define e_cut and cut values on neutrino prob
    e_cuts = ((1, 100), (1, 5), (5, 10), (10, 20), (20, 100))
    cuts = np.logspace(-5.9, 1, num=10000)  # cuts = np.linspace(0, 1, 10000)
    prob_not_neutrino = pred_dl['prob_not_neutrino']
    muon_score_std = pred_std['prob_muon']
    rn_noise_score_std = pred_std['prob_noise']
    w_1_y = mc_info_dl['oscillated_weight_one_year_bg_sel']
    w_1_y_std = mc_info_std['oscillated_weight_one_year_bg_sel']

    for tpl in e_cuts:
        print('Making contamination plots for e-cut ' + str(tpl))

        e_low, e_high = tpl[0], tpl[1]
        e_cut_mask_dl = np.logical_and(mc_info_dl['energy'] >= e_low, mc_info_dl['energy'] < e_high)
        e_cut_mask_std = np.logical_and(mc_info_std['energy'] >= e_low, mc_info_std['energy'] < e_high)

        # make energy cuts on neutrinos only
        is_neutrino_e_cut_dl = np.logical_and(e_cut_mask_dl, is_neutrino_dl)
        is_neutrino_e_cut_std = np.logical_and(e_cut_mask_std, is_neutrino_std)

        # n_neutrinos_total_e_cut = np.count_nonzero(is_neutrino_e_cut_dl)  # should be same for both dl and std
        n_neutrinos_total_e_cut_weighted = np.sum(w_1_y[is_neutrino_e_cut_dl])

        n_neutrinos_total_e_cut_weighted_std = np.sum(w_1_y_std[is_neutrino_e_cut_std])
        assert np.round(n_neutrinos_total_e_cut_weighted_std, 5) == np.round(n_neutrinos_total_e_cut_weighted, 5)

        # x_contamination_mupage_dl, y_neutrino_efficiency_dl, y_n_muons_dl = [], [], []
        # x_contamination_mupage_std, y_neutrino_efficiency_std, y_n_muons_std = [], [], []

        neutrino_eff_weighted_dl, muon_cont_weighted_dl, rn_cont_weighted_dl = [], [], []
        neutrino_eff_weighted_std, muon_cont_weighted_std, rn_cont_weighted_std = [], [], []

        muon_cont_weighted_dl_err, muon_cont_weighted_std_err = [], []
        rn_cont_weighted_dl_err, rn_cont_weighted_std_err = [], []

        for i in range(len(cuts)):

            # dl

            # if np.count_nonzero(mupage_sel_dl) != 0:
            #     fract_n_mupage_err_to_n_mupage_dl = n_mupage_err_dl / np.count_nonzero(mupage_sel_dl)
            # else:
            #     fract_n_mupage_err_to_n_mupage_dl = 0

            neutr_sel_dl = prob_not_neutrino[is_neutrino_e_cut_dl] < cuts[i]
            n_neutrino_weighted_dl = np.sum(w_1_y[is_neutrino_e_cut_dl][neutr_sel_dl])

            if n_neutrino_weighted_dl != 0:
                mupage_sel_dl = prob_not_neutrino[is_mupage_dl] < cuts[i]
                rn_sel_dl = prob_not_neutrino[is_rn_dl] < cuts[i]

                n_mupage_weighted_dl = np.sum(w_1_y[is_mupage_dl][mupage_sel_dl])
                n_rn_weighted_dl = np.sum(w_1_y[is_rn_dl][rn_sel_dl])

                neutrino_eff_weighted_dl.append((n_neutrino_weighted_dl / n_neutrinos_total_e_cut_weighted) * 100)
                muon_cont_weighted_dl.append((n_mupage_weighted_dl / n_neutrino_weighted_dl) * 100)
                rn_cont_weighted_dl.append((n_rn_weighted_dl / n_neutrino_weighted_dl) * 100)

                # we calculate n_mupage_weighted_dl / n_neutrino_weighted_dl
                # errors on n_neutrino can be neglected, since statistics large
                # gaussian error prop: err_cont = abs(1/n_neutr) * delta n_mupage

                n_mupage_err_dl = np.sqrt(np.count_nonzero(mupage_sel_dl))
                n_rn_err_dl = np.sqrt(np.count_nonzero(rn_sel_dl))
                # n_mupage_weighted_err_dl = n_mupage_weighted_dl * fract_n_mupage_err_to_n_mupage_dl
                # muon_cont_weighted_dl_err.append((np.abs(1/np.count_nonzero(neutr_sel_dl)) * n_mupage_err_dl) * 100)
                muon_cont_weighted_dl_err.append(((n_mupage_err_dl * 33.2137) / n_neutrino_weighted_dl) * 100)
                rn_cont_weighted_dl_err.append(((n_rn_err_dl * 332.87896624) / n_neutrino_weighted_dl) * 100)

            # std

            # if np.count_nonzero(mupage_sel_std) != 0:
            #     fract_n_mupage_err_to_n_mupage_std = n_mupage_err_std / np.count_nonzero(mupage_sel_std)
            # else:
            #     fract_n_mupage_err_to_n_mupage_std = 0

            neutr_sel_std = muon_score_std[is_neutrino_e_cut_std] < cuts[i]
            n_neutrino_weighted_std = np.sum(w_1_y_std[is_neutrino_e_cut_std][neutr_sel_std])

            if n_neutrino_weighted_std != 0:
                mupage_sel_std = muon_score_std[is_mupage_std] < cuts[i]
                rn_sel_std = rn_noise_score_std[is_rn_std] < cuts[i]

                n_mupage_weighted_std = np.sum(w_1_y_std[is_mupage_std][mupage_sel_std])
                n_rn_weighted_std = np.sum(w_1_y_std[is_rn_std][rn_sel_std])

                neutrino_eff_weighted_std.append((n_neutrino_weighted_std / n_neutrinos_total_e_cut_weighted) * 100)
                muon_cont_weighted_std.append((n_mupage_weighted_std / n_neutrino_weighted_std) * 100)
                rn_cont_weighted_std.append((n_rn_weighted_std / n_neutrino_weighted_std) * 100)

                # we calculate n_mupage_weighted_std / n_neutrino_weighted_std
                # errors on n_neutrino can be neglected, since statistics large
                # gaussian error prop: err_cont = abs(1/n_neutr) * delta n_mupage

                n_mupage_err_std = np.sqrt(np.count_nonzero(mupage_sel_std))
                n_rn_err_std = np.sqrt(np.count_nonzero(rn_sel_std))
                # n_mupage_weighted_err_std = n_mupage_weighted_std * fract_n_mupage_err_to_n_mupage_std
                # muon_cont_weighted_std_err.append((np.abs(1/np.count_nonzero(neutr_sel_std)) * n_mupage_err_std) * 100)
                muon_cont_weighted_std_err.append(((n_mupage_err_std * 33.2137) / n_neutrino_weighted_std) * 100)
                rn_cont_weighted_std_err.append(((n_rn_err_std * 332.87896624) / n_neutrino_weighted_std) * 100)

        # remove all 0 entries with n_muons, only needed for dl, since standard reco cant go to 0 muons anyways
        # idx_last_zero = np.amax(np.where(np.array(y_n_muons_dl) == 0))
        # y_n_muons_dl = y_n_muons_dl[idx_last_zero:]
        # y_neutrino_efficiency_dl = y_neutrino_efficiency_dl[idx_last_zero:]
        # x_contamination_mupage_dl = x_contamination_mupage_dl[idx_last_zero:]

        make_plots()
