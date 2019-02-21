#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making plots for background classifiers (muon/random_noise/neutrinos).
"""

from matplotlib import pyplot as plt
import numpy as np
from orcanet_contrib.plotting.utils import get_event_selection_mask


def make_prob_hists_bg_classifier(pred_file, savefolder, cuts=None):
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

    """
    def configure_and_save_plot(bg_class, savefolder):
        """
        Configure and save a mpl plot with GridLines, Logscale etc. for the background classifier probability plots.
        Parameters
        ----------
        bg_class : str
            A string which specifies the probability class: (muon, random_noise, neutrino).
        savefolder : str
            Path to the directory, where the plot should be saved.

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

        plt.savefig(savefolder + '/prob_' + bg_class + '.pdf')
        plt.savefig(savefolder + '/prob_' + bg_class + '.png', dpi=600)

    fig, axes = plt.subplots()

    #bg_classes = ['muon', 'random_noise', 'neutrino']
    bg_classes = ['not_neutrino', 'neutrino']
    for bg_class in bg_classes:
        prob_class = pred_file['pred']['prob_' + bg_class]
        ptype, is_cc = pred_file['mc_info']['particle_type'], pred_file['mc_info']['is_cc']

        if cuts is not None:
            assert isinstance(cuts, str)
            evt_sel_mask = get_event_selection_mask(pred_file['mc_info'], cut_name=cuts)
            prob_class = prob_class[evt_sel_mask]
            ptype, is_cc = ptype[evt_sel_mask], is_cc[evt_sel_mask]

        make_prob_hists_for_class(prob_class, ptype, is_cc, axes)
        configure_and_save_plot(bg_class, savefolder)
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


def make_prob_hists_for_class(prob_class, ptype, is_cc, axes):
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

    axes.hist(prob_class[is_mupage], density=True, bins=40, range=(0, 1), color='b', label='mupage', histtype='step', linestyle='-', zorder=3)
    axes.hist(prob_class[is_random_noise], density=True, bins=40, range=(0, 1), color='r', label='random_noise', histtype='step', linestyle='-', zorder=3)
    axes.hist(prob_class[is_neutrino], density=True, bins=40, range=(0, 1), color='saddlebrown', label='neutrino', histtype='step', linestyle='-', zorder=3)
