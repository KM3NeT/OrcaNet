#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making plots for track-shower classifiers.
"""

import numpy as np
from matplotlib import pyplot as plt
from orcanet_contrib.plotting.utils import get_event_selection_mask, select_ic


# -- Utility functions -- #

def get_latex_code_for_ptype_str(class_name):
    """
    Returns the mpl latex code for a certain particle type class (e.g. muon-CC).

    Parameters
    ----------
    class_name : str
        String that specifies the particle type class.

    Returns
    -------
    label : str
        class_name, reformated into a latex readable string.

    """
    labels = {'muon-CC': r'$\nu_{\mu}-CC$', 'a_muon-CC': r'$\overline{\nu}_{\mu}-CC$', 'elec-CC': r'$\nu_{e}-CC$',
              'a_elec-CC': r'$\overline{\nu}_{e}-CC$', 'elec-NC': r'$\nu_{e}-NC$',
              'a_elec-NC': r'$\overline{\nu}_{e}-NC$', 'tau-CC': r'$\nu_{\tau}-CC$',
              'a_tau-CC': r'$\overline{\nu}_{\tau}-CC$'}
    label = labels[class_name]

    return label

# -- Utility functions -- #


# -- Functions for making energy to accuracy plots -- #

def make_e_to_acc_plot_ts(pred_file, title, savefolder, plot_range=(1, 100),
                          cuts=None, prob_threshold_shower=0.5):
    """
    Makes a track-shower step plot for energy to "Fraction of events classified as track" for different neutrino types.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the track-shower classification predictions of a nn model.
    title : str
        Title that should be used for the plot.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    plot_range : tuple(int,int)
        Tuple that specifies the X-Range of the plot.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.
    prob_threshold_shower : float
        Sets the lower threshold for when an event is classified as a shower based on the nn shower probability.

    """
    y_pred, y_true, mc_info = pred_file['pred'], pred_file['true'], pred_file['mc_info']
    nn_pred_correct = get_nn_pred_correct_info(y_pred, y_true, prob_threshold_shower=prob_threshold_shower)
    print_accuracy(nn_pred_correct, print_text='Accuracy of the T/S classifier without any event selection: ')

    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        nn_pred_correct = nn_pred_correct[evt_sel_mask]
        print_accuracy(nn_pred_correct, print_text='Accuracy of the T/S classifier with the event selection "'
                                                   + cuts + '": ')

    fig, axes = plt.subplots()

    make_step_plot_e_acc_class('muon-CC', mc_info, nn_pred_correct, axes, plot_range=plot_range, linestyle='-', color='b')
    make_step_plot_e_acc_class('a_muon-CC', mc_info, nn_pred_correct, axes, plot_range=plot_range, linestyle='--', color='b')
    make_step_plot_e_acc_class('elec-CC', mc_info, nn_pred_correct, axes, invert=True, plot_range=plot_range, linestyle='-', color='r')
    make_step_plot_e_acc_class('a_elec-CC', mc_info, nn_pred_correct, axes, invert=True, plot_range=plot_range, linestyle='--', color='r')
    make_step_plot_e_acc_class('elec-NC', mc_info, nn_pred_correct, axes, invert=True, plot_range=plot_range, linestyle='-', color='saddlebrown')
    make_step_plot_e_acc_class('a_elec-NC', mc_info, nn_pred_correct, axes, invert=True, plot_range=plot_range, linestyle='--', color='saddlebrown')
    make_step_plot_e_acc_class('tau-CC', mc_info, nn_pred_correct, axes, invert=True, plot_range=plot_range, linestyle='-', color='g')
    make_step_plot_e_acc_class('a_tau-CC', mc_info, nn_pred_correct, axes, invert=True, plot_range=plot_range, linestyle='--', color='g')

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

    plt.savefig(savefolder + '/ts_e_to_acc_3-100GeV.pdf')
    plt.savefig(savefolder + '/ts_e_to_acc_3-100GeV.png', dpi=600)

    x_ticks_major = np.arange(0, 101, 5)
    plt.xticks(x_ticks_major)
    plt.xlim((0, 40))
    plt.savefig(savefolder + '/ts_e_to_acc_3-40GeV.pdf')
    plt.savefig(savefolder + '/ts_e_to_acc_3-40GeV.png', dpi=600)

    plt.close()


def get_nn_pred_correct_info(y_pred, y_true, prob_threshold_shower=0.5):
    """
    Function that checks if the predictions by the neural network are correct or not for the binary T/S classification.

    Parameters
    ----------
    y_pred : ndarray(ndim=2)
        Structured array with the nn probabilities for the shower and the track class.
    y_true : ndarray(ndim=2)
        Structured array with the one hot true values for the shower and the track class.
    prob_threshold_shower : float
        Lower threshold, which sets the minimum value, at which an event is considered a shower based
        on the predicted nn shower probability.

    Returns
    -------
    nn_pred_correct : ndarray(ndim=1)
        Boolean array which specifies, if the predictions of the nn for each event are correct or not.

    """
    if len(y_pred.dtype) > 2:
        raise ValueError('The check if a T/S prediction of a nn is correct is only available for '
                         'binary categorization problems and not for problems with more than two classes!')

    is_shower_pred = (y_pred['prob_shower'] > prob_threshold_shower)
    is_shower_true = (y_true['cat_shower'] > 0)

    # True: if True, True (correct pred for shower) or False, False (correct pred for track)
    nn_pred_correct = is_shower_pred == is_shower_true

    return nn_pred_correct


def print_accuracy(nn_pred_correct, print_text='Accuracy of the T/S classifier: '):
    """
    Prints the T/S accuracy of a nn based on the nn_pred_correct info.

    Parameters
    ----------
    nn_pred_correct : ndarray(ndim=1)
        Boolean array which specifies, if the predictions of the nn for each event are correct or not.

    print_text : str
        String that should be used in printing before printing the results.

    """
    n_correct = np.count_nonzero(nn_pred_correct)
    n_total = nn_pred_correct.shape[0]
    accuracy = n_correct / float(n_total)

    print(print_text + '\n' + str(accuracy * 100) + ', based on ' + str(n_total) + ' events')


def select_class(class_name, ptype, is_cc):
    """
    Returns a boolean array which specifies, which rows in the ptype & is_cc 1d arrays belong to a single
    class, specified by the class_name.

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
    particle_types = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1),
                      'a_elec-CC': (-12, 1), 'elec-NC': (12, 0), 'a_elec-NC': (-12, 0),
                      'tau-CC': (16, 1), 'a_tau-CC': (-16, 1)}

    try:
        is_class = np.logical_and(ptype == particle_types[class_name][0], is_cc == particle_types[class_name][1])
    except KeyError:
        raise ValueError('The class ' + str(class_name) + ' is not available in the particle_types dictionary.')

    return is_class


def make_step_plot_e_acc_class(class_name, mc_info, nn_pred_correct, axes, invert=False, plot_range=(3, 100),
                               linestyle='-', color='b'):
    """
    Plots 1D "energy" to "Accuracy" step plots to an axes object for a certain particle type class (e.g. a_muon-CC)

    Parameters
    ----------
    class_name : str
        Particle type that should be plotted, e.g. a_muon-CC.
    mc_info : ndarray(ndim=2)
        Structured array containing the MC info.
    nn_pred_correct : ndarray(ndim=1)
        Boolean array which specifies, if the predictions of the nn for each event are correct or not.
    axes : mpl.axes
        Matplotlib axes object.
    invert : bool
        If True, it inverts the y-axis which may be useful for plotting a
        'Fraction of events classified as track' plot.
    plot_range : tuple(int,int)
        Tuple that specifies the X-Range of the plot.
    linestyle : str
         Specifies the mpl linestyle that should be used.
    color : str
        Specifies the mpl color that should be used for plotting the step plot.

    """
    ptype, is_cc = mc_info['particle_type'], mc_info['is_cc']

    is_class = select_class(class_name, ptype, is_cc)

    energy = mc_info[is_class]['energy']
    if energy.size == 0:
        return  # class not contained
    nn_pred_correct = nn_pred_correct[is_class].astype(np.int8)

    hist_1d_energy = np.histogram(energy, bins=99, range=plot_range)
    # e-hist for only correctly classified events
    hist_1d_energy_pred_correct = np.histogram(energy[nn_pred_correct == 1], bins=99, range=plot_range)

    bin_edges = hist_1d_energy[1]
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_pred_correct[0], hist_1d_energy[0], dtype=np.float64)  # TODO solve division by zero

    # This function makes plots with a "Fraction of events classified as track" Y-axis
    # , so need to invert the hist_1d_energy_accuracy for shower events!
    if invert is True:
        hist_1d_energy_accuracy_bins = np.absolute(hist_1d_energy_accuracy_bins - 1)

    # For making it work with matplotlib step plot
    hist_1d_energy_accuracy_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_bins))

    label = get_latex_code_for_ptype_str(class_name)

    axes.step(bin_edges, hist_1d_energy_accuracy_bins_leading_zero, where='pre', linestyle=linestyle, color=color,
              label=label, zorder=3)

# -- Functions for making energy to accuracy plots -- #


# -- Functions for making probability plots -- #

def make_ts_prob_hists(pred_file, savefolder, cuts=None):
    """
    Makes track-shower histograms for the nn class predictions (track, shower).

    In total, 2 plots will be generated:
    1) A plot which shows the nn prediction probabilities for the track class for each particle type class
    1) A plot which shows the nn prediction probabilities for the shower class for each particle type class

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the track-shower classification predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.

    """
    def configure_and_save_plot(plot_title, savepath):
        """
        Configure a mpl plot with GridLines, Logscale etc.

        Parameters
        ----------
        plot_title : str
            Title that should be used for the plot.
        savepath : str
            Filepath that should be used for saving the plot.

        """
        axes.legend(loc='upper center', ncol=2)
        plt.grid(True, zorder=0, linestyle='dotted')
        # plt.yscale('log')

        x_ticks_major = np.arange(0, 1.1, 0.1)
        plt.xticks(x_ticks_major)
        plt.minorticks_on()

        plt.xlabel('OrcaNet probability')
        plt.ylabel('Normed Quantity')
        title = plt.title(plot_title)
        title.set_position([.5, 1.04])

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png', dpi=600)

    fig, axes = plt.subplots()

    # make energy cut, 1-40GeV
    y_pred, mc_info = pred_file['pred'], pred_file['mc_info']
    e_lower_40 = mc_info['energy'] <= 40
    y_pred, mc_info = y_pred[e_lower_40], mc_info[e_lower_40]

    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        y_pred = y_pred[evt_sel_mask]

    for ts_class in ['track', 'shower']:
        make_prob_hist_class('muon-CC', mc_info, y_pred, ts_class, axes, color='b', linestyle='-')
        make_prob_hist_class('a_muon-CC', mc_info, y_pred, ts_class, axes, color='b', linestyle='--')
        make_prob_hist_class('elec-CC', mc_info, y_pred, ts_class, axes, color='r', linestyle='-')
        make_prob_hist_class('a_elec-CC', mc_info, y_pred, ts_class, axes, color='r', linestyle='--')
        make_prob_hist_class('elec-NC', mc_info, y_pred, ts_class, axes, color='saddlebrown', linestyle='-')
        make_prob_hist_class('a_elec-NC', mc_info, y_pred, ts_class, axes, color='saddlebrown', linestyle='--')
        make_prob_hist_class('tau-CC', mc_info, y_pred, ts_class, axes, color='g', linestyle='-')
        make_prob_hist_class('a_tau-CC', mc_info, y_pred, ts_class, axes, color='g', linestyle='--')
        configure_and_save_plot(plot_title='Probability to be classified as ' + ts_class + ', 3-40GeV',
                                savepath=savefolder + '/ts_prob_' + ts_class)
        plt.cla()

    plt.close()


def make_prob_hist_class(class_name, mc_info, y_pred, ts_class, axes, plot_range=(0, 1), color='b', linestyle='-'):
    """
    Adds a mpl ts probability hist to an axes object for a certain particle class (e.g. 'muon-CC').

    The X-axis of the hist is the probability for the ts_class, and the Y-axis shows the normed quantity.

    Parameters
    ----------
    class_name : str
        Particle type that should be plotted in the hist, e.g. a_muon-CC.
    mc_info : ndarray(ndim=2)
        Structured array containing the MC info.
    y_pred : ndarray(ndim=2)
        Structured array containing the predicted nn probabilities for the shower and the track class.
    ts_class : str
        Specifies for which ts class ("shower", "track") the nn prediction probability should be plotted.
    axes : mpl.axes
        Matplotlib axes object.
    plot_range : tuple(int,int)
        Tuple that specifies the X-Range of the plot.
    color : str
        Specifies the mpl color that should be used for plotting the hist.
    linestyle : str
        Specifies the mpl linestyle that should be used.

    """
    ptype, is_cc = mc_info['particle_type'], mc_info['is_cc']
    is_class = select_class(class_name, ptype, is_cc)

    nn_prob_class = y_pred[is_class]['prob_' + ts_class]
    if nn_prob_class.size == 0:
        return  # class not contained

    label = get_latex_code_for_ptype_str(class_name)

    axes.hist(nn_prob_class, bins=40, range=plot_range, density=True, color=color, label=label,
              histtype='step', linestyle=linestyle, zorder=3)

# -- Functions for making probability plots -- #


def plot_ts_separability(pred_file, savefolder, pred_file_2=None, cuts=None):
    """
    Calculates and plots the separability (1-correlation coefficient) plot for ts classifiers.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the track-shower classification predictions of a nn model.
    savefolder : str
        Path of the directory, where the plots should be saved to.
    pred_file_2 : None/h5py.File
        H5py file instance of the predictions of a second classifier, that should be plotted
        into the same plot, e.g. a shallow learning classifier as comparison.
    cuts : None/str
        Specifies, if cuts should be used for the plot. Either None or a str, that is available in the
        load_event_selection_file() function.

    """
    mc_info = pred_file['mc_info']
    prob_track = pred_file['pred']['prob_track']

    if cuts is not None:
        assert isinstance(cuts, str)
        evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
        mc_info = mc_info[evt_sel_mask]
        prob_track = prob_track[evt_sel_mask]

    separabilities = calculcate_separability(mc_info, prob_track)

    # plot the correlation coefficients
    fig, axes = plt.subplots()
    plt.plot(separabilities[:, 1], separabilities[:, 0], 'b', marker='o', lw=0.5, markersize=3,
             label='Deep Learning')

    if pred_file_2 is not None:
        mc_info_2 = pred_file_2['mc_info']
        prob_track_2 = pred_file_2['pred']['prob_track']

        if cuts is not None:
            assert isinstance(cuts, str)
            evt_sel_mask_2 = get_event_selection_mask(mc_info_2, cut_name=cuts)
            mc_info_2 = mc_info_2[evt_sel_mask_2]
            prob_track_2 = prob_track_2[evt_sel_mask_2]

        separabilities_2 = calculcate_separability(mc_info_2, prob_track_2)
        plt.plot(separabilities_2[:, 1], separabilities_2[:, 0], 'r', marker='o', lw=0.5,
                 markersize=3, label='Shallow Learning')

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

    plt.savefig(savefolder + '/ts_correlation_coefficients.pdf')
    plt.savefig(savefolder + '/ts_correlation_coefficients.png', dpi=600)

    plt.close()


def calculcate_separability(mc_info, prob_track, bins=40, e_cut_range=np.logspace(0.3, 2, 18)):
    """
    Calculates the separability per energy bin by calculating the correlation factors (s=1-c).

    Parameters
    ----------
    mc_info : h5py.dataset.Dataset/ndarray(ndim=2)
        The mc_info dataset of an OrcaNet nn prediction file.
    prob_track : ndarray(ndim=1)
        Array containing the track probabilities by a classifier.
    bins : int
        For how many energy bins the separability should be calculated.
    e_cut_range : ndarray(ndim=1)
        Energy range for the calculation of the separability factors.

    """
    mc_energy = mc_info['energy']
    particle_type, is_cc = mc_info['particle_type'], mc_info['is_cc']
    is_muon_cc = select_ic(particle_type, is_cc, 'muon-CC')
    is_elec_cc = select_ic(particle_type, is_cc, 'elec-CC')

    n = 0
    separabilities = []
    for e_cut in zip(e_cut_range[:-1], e_cut_range[1:]):
        n += 1
        if n <= 2:
            continue  # ecut steffen

        e_cut_mask = np.logical_and(e_cut[0] <= mc_energy, mc_energy <= e_cut[1])

        is_muon_cc_and_e_cut = np.logical_and(is_muon_cc, e_cut_mask)
        is_elec_cc_and_e_cut = np.logical_and(is_elec_cc, e_cut_mask)

        hist_prob_track_e_cut_muon_cc = np.histogram(prob_track[is_muon_cc_and_e_cut], bins=40, density=True)
        hist_prob_track_e_cut_elec_cc = np.histogram(prob_track[is_elec_cc_and_e_cut], bins=40, density=True)

        correlation_coeff_enumerator = 0
        for j in range(bins - 1):
            correlation_coeff_enumerator += hist_prob_track_e_cut_muon_cc[0][j] * hist_prob_track_e_cut_elec_cc[0][j]

        sum_prob_muon_cc = np.sum(hist_prob_track_e_cut_muon_cc[0] ** 2)
        sum_prob_elec_cc = np.sum(hist_prob_track_e_cut_elec_cc[0] ** 2)

        correlation_coeff_denominator = np.sqrt(sum_prob_muon_cc * sum_prob_elec_cc)

        separability = 1 - correlation_coeff_enumerator / float(correlation_coeff_denominator)

        average_energy = 10 ** ((np.log10(e_cut[1]) + np.log10(e_cut[0])) / float(2))

        separabilities.append((separability, average_energy))

    separabilities = np.array(separabilities)

    return separabilities
