#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code for making nn reco performance plots.
"""

import numpy as np


def select_track_shower(ptype, is_cc):
    """
    Returns a boolean mask which specifies, which events in the rows of the particle_type and is_cc arrays belong to
    the track and which to the shower class.

    Parameters
    ----------
    ptype : ndarray(ndim=1)
        Array with particle_types of some events.
    is_cc : ndarray(ndim=1)
        Array with is_cc of some events.

    Returns
    -------
    is_track : ndarray(ndim=1)
        Boolean array which specifies, if the events from the ptype & is_cc arrays belong to the track class or not.
    is_shower : ndarray(ndim=1)
        Boolean array which specifies, if the events from the ptype & is_cc arrays belong to the shower class or not.

    """
    abs_particle_type = np.abs(ptype)
    is_track = np.logical_and(abs_particle_type == 14, is_cc == 1)
    is_shower = np.logical_or(abs_particle_type == 16, abs_particle_type == 12)

    return is_track, is_shower


def select_ic(ptype, is_cc, interaction_channel):
    """
    Returns a boolean mask which specifies, which events in the rows of the particle_type and is_cc arrays belong to
    the specified interaction_channel.

    Parameters
    ----------
    ptype : ndarray(ndim=1)
        Array with particle_types of some events.
    is_cc : ndarray(ndim=1)
        Array with is_cc of some events.
    interaction_channel : str
        String, that specifies the interaction channel (one of "muon-CC", "elec-CC", "elec-NC", "tau-CC")

    Returns
    -------
    is_ic : ndarray(ndim=1)
        Boolean mask array which specifies if each event belongs to the input ic or not.

    """
    ic_dict = {'muon-CC': (14, 1), 'elec-CC': (12, 1), 'elec-NC': (12, 0), 'tau-CC': (16, 1)}
    if interaction_channel not in list(ic_dict.keys()):
        raise ValueError('The interaction_channel ' + str(interaction_channel) + ' is not known.')

    is_ic = np.logical_and(np.abs(ptype) == ic_dict[interaction_channel][0],
                           is_cc == ic_dict[interaction_channel][1])

    return is_ic


def correct_reco_energy(mc_info, energy_pred_array, metric='median'):
    """
    Makes the correction factors based on e-CC events, applies them to ALL shower events (e-CC, e-NC, tau-CC) and
    returns the corrected pred_energy array.

    Parameters
    ----------
    mc_info : h5py.dataset.Dataset/ndarray(ndim=2)
        The mc_info structured array of an OrcaNet nn prediction file.
    energy_pred_array : ndarray(ndim=1)
        Array, which contains the predicted energies of a nn.
    metric : str
        Metric that should be used for the correction. Available: 'median' and 'mean'.

    Returns
    -------
    energy_pred_corrected : ndarray(ndim=1)
        Array containing the corrected prediction energies.

    """
    particle_type, is_cc = mc_info['particle_type'], mc_info['is_cc']

    is_track, is_shower = select_track_shower(particle_type, is_cc)
    is_ic = select_ic(particle_type, is_cc, 'elec-CC')

    energy_mc_e_cc, energy_pred_e_cc = mc_info['energy'][is_ic], energy_pred_array[is_ic]

    correction_factors_x, correction_factors_y = [], []

    e_range = np.logspace(np.log(3)/np.log(2), np.log(100)/np.log(2), 50, base=2)
    n_ranges = e_range.shape[0] - 1
    for i in range(n_ranges):
        e_range_low, e_range_high = e_range[i], e_range[i+1]
        e_range_mean = (e_range_high + e_range_low) / float(2)

        e_mc_cut_boolean = np.logical_and(e_range_low < energy_mc_e_cc, energy_mc_e_cc <= e_range_high)
        e_mc_cut = energy_mc_e_cc[e_mc_cut_boolean]
        e_pred_cut = energy_pred_e_cc[e_mc_cut_boolean]

        if metric == 'median':
            correction_factor = np.median((e_pred_cut - e_mc_cut) / e_pred_cut)
        elif metric == 'mean':
            correction_factor = np.mean((e_pred_cut - e_mc_cut) / e_pred_cut)
        else:
            raise ValueError('The specified metric "' + metric + '" is not implemented.')

        correction_factors_x.append(e_range_mean)
        correction_factors_y.append(correction_factor)

    # linear interpolation of correction factors
    energy_pred_orig_shower = energy_pred_array[is_shower]
    correction_factor_en_pred = np.interp(energy_pred_orig_shower, correction_factors_x, correction_factors_y)

    # apply correction to ALL shower ic's (including all taus atm)
    # need to make a copy of the array, we dont want to make changes in the original array
    energy_pred_corrected = np.copy(energy_pred_array)
    energy_pred_corrected[is_shower] = energy_pred_orig_shower + (- correction_factor_en_pred) * energy_pred_orig_shower

    return energy_pred_corrected


# --------------------------- Code for making cuts --------------------------- #

def get_event_selection_mask(mc_info, invert=False, cut_name='neutrino_regr'):
    """
    Function that checks, which events in the mc_info input array are also in the event selection file,
    specified by the cut_name parameter.

    Parameters
    ----------
    mc_info : ndarray(ndim=2)
        The mc_info structured array of an OrcaNet nn prediction file.
    invert : Instead of selecting all events that survive some cuts, this _removes_ all the events that are normally
             cut and only leaves the events that are not contained in the cut_file specified by the cut_name.
    cut_name : str
        String, which specifies the actual cut file that should be loaded.
    Returns
    -------
    bool_evt_selected : ndarray(ndim=1)
        Boolean array, which specifies which events in the mc_info are contained in the cut_file specified by
        the cut_name parameter.

    """
    # load array with run_id, event_id, prod_ident, particle_type and is_cc info (in that column order!)
    arr_sel_events = load_event_selection_file(cut_name)

    # select the same information from the input mc_info
    ax = np.newaxis
    mc_info_necessary_info = np.concatenate([mc_info['run_id'][:, ax], mc_info['event_id'][:, ax],
                                             mc_info['prod_ident'][:, ax], mc_info['particle_type'][:, ax],
                                             mc_info['is_cc'][:, ax]], axis=1)

    bool_evt_selected = in_nd(mc_info_necessary_info, arr_sel_events)

    if invert is True:
        bool_evt_selected = np.invert(bool_evt_selected)

    return bool_evt_selected


def load_event_selection_file(cut_name, dirpath='/home/saturn/capn/mppi033h/Data/event_selections'):
    """
    Loads a .npy file base on the cut_name and the dirpath, which contains the event selection info,
    e.g. of some other reco.

    Parameters
    ----------
    cut_name : str
        String, which specifies the actual cut file that should be loaded.
    dirpath : str
        Path to the directory, where the file specified by the cut_name is located.

    Returns
    -------
    selected_events_array : ndarray(ndim=2)
        Array with the run_id, event_id, prod_ident, particle_type and is_cc info of the selected events.

    """
    if cut_name == 'neutrino_regr':
        # contains run_id, event_id, prod_ident, particle_type and is_cc
        selected_events_array = np.load(dirpath + '/evt_selection_regression.npy')

    elif cut_name == 'neutrino_ts':
        # contains run_id, event_id, prod_ident, particle_type and is_cc
        selected_events_array = np.load(dirpath + '/evt_selection_ts_classifier.npy')

    elif cut_name == 'bg_classifier':
        # contains run_id, event_id, prod_ident, particle_type and is_cc
        selected_events_array = np.load(dirpath + '/evt_selection_bg_classifier.npy')

    else:
        raise ValueError('The specified cut_name "' + str(cut_name) + '" is not available.')

    return selected_events_array


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
    Function that generalizes the numpy in_1d function to nd. The docs are specifically for the 2d case though.

    Checks if entries in axis_0 of a exist in b and returns the bool array for all rows.

    Parameters
    ----------
    a : ndarray(ndim=2)
        Array where it should be checked whether each row exists in b or not.
    b : ndarray(ndim>=2)
        Array upon which the rows of a are checked.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which can speed up the calculation.

    Returns
    -------
     a_in_b: ndarray(ndim=1)
        Boolean array that specifies for each row of a if it also exists in b or not.

    """
    a = asvoid(a)
    b = asvoid(b)
    a_in_b = np.in1d(a, b, assume_unique)
    return a_in_b


def test_in_nd():
    a = np.array([[1, 129385, 1, -1, -0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    b = np.array([[1, 1, 0, 0, 2], [6, 6, 6, 6, 6], [1, 129385, 1, -1, -0]])

    c = in_nd(a, b)
    it_works = np.all(c == [True, False, False])
    return it_works


def get_cut_mask_for_events_that_exist_in_both_files(pred_file_1, pred_file_2, remove_duplicates=(False, False)):
    """

    """
    ax = np.newaxis
    # select pred_file_1 evts that are also in pred_file_2
    print('Shape of mc_info pred_file_1 before selection: ' + str(pred_file_1['mc_info'].shape))
    print('Shape of mc_info pred_file_2 before selection: ' + str(pred_file_2['mc_info'].shape))

    mc_info_1, mc_info_2 = pred_file_1['mc_info'], pred_file_2['mc_info']

    mc_info_id_1 = np.concatenate([mc_info_1['run_id'][:, ax], mc_info_1['event_id'][:, ax],
                                   mc_info_1['prod_ident'][:, ax], mc_info_1['particle_type'][:, ax],
                                   mc_info_1['is_cc'][:, ax]], axis=1)
    mc_info_id_2 = np.concatenate([mc_info_2['run_id'][:, ax], mc_info_2['event_id'][:, ax],
                                   mc_info_2['prod_ident'][:, ax], mc_info_2['particle_type'][:, ax],
                                   mc_info_2['is_cc'][:, ax]], axis=1)

    # boolean mask for file 1 that specifies, if events in 1 have been found in 2
    mask_1_in_2 = in_nd(mc_info_id_1, mc_info_id_2)
    mc_info_id_1_sel = mc_info_id_1[mask_1_in_2]

    mask_2_in_1_sel = in_nd(mc_info_id_2, mc_info_id_1_sel)

    print('Shape of mc_info pred_file_1 after selection: ' + str(mc_info_id_1_sel.shape))
    print('Shape of mc_info pred_file_2 after selection: ' + str(mc_info_id_2[mask_2_in_1_sel].shape))

    if remove_duplicates[0]:
        # get rid of duplicates!
        print('Shape of cut mc_info pred_file_1 before deselecting the few duplicates: ' + str(mc_info_id_1_sel.shape))
        unq_1, idx_1, count_1 = np.unique(mc_info_id_1, axis=0,
                                          return_counts=True, return_index=True)
        # select only unique rows
        mask_uq_1 = np.zeros(mc_info_id_1.shape[0], np.bool)
        mask_uq_1[idx_1] = 1

        mask_1_in_2_uq = np.logical_and(mask_1_in_2, mask_uq_1)

        print('Shape of cut mc_info pred_file_1 after deselecting the few duplicates: ' + str(mc_info_id_1[mask_1_in_2_uq].shape))

        mask_1_in_2 = mask_1_in_2_uq

    if remove_duplicates[1]:
        # get rid of duplicates!
        print('Shape of cut mc_info pred_file_2 before deselecting the few duplicates: ' + str(mc_info_id_2[mask_2_in_1_sel].shape))
        unq_2, idx_2, count_2 = np.unique(mc_info_id_2, axis=0,
                                          return_counts=True, return_index=True)
        # select only unique rows
        mask_uq_2 = np.zeros(mc_info_id_2.shape[0], np.bool)
        mask_uq_2[idx_2] = 1

        mask_2_in_1_sel_uq = np.logical_and(mask_2_in_1_sel, mask_uq_2)

        print('Shape of cut mc_info pred_file_2 after deselecting the few duplicates: ' + str(
              mc_info_id_2[mask_2_in_1_sel_uq].shape))

        mask_2_in_1_sel = mask_2_in_1_sel_uq

    return mask_1_in_2, mask_2_in_1_sel


def get_mc_info_and_other_datasets(pred_file, mc_info_key, dset_keys, cuts=None):
    """
    Gets the mc_info and pred_datasets from a h5py pred_file instance and applies some cuts if cuts is not None.

    Parameters
    ----------
    pred_file : h5py.File
        H5py file instance, which stores the regression predictions of a nn model.
    mc_info_key : str
        Key of the mc_info dataset in the pred_file.
    dset_keys : tuple/str
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
    if not isinstance(dset_keys, tuple):
        dset_keys = (dset_keys,)

    preds = []
    mc_info = pred_file[mc_info_key]
    if cuts is not None:
        if type(cuts) is np.ndarray:
            mc_info = mc_info[cuts]
            for key in dset_keys:
                preds.append(pred_file[key][cuts])

        else:
            assert isinstance(cuts, str)
            print('Event number before selection: ' + str(mc_info.shape))
            evt_sel_mask = get_event_selection_mask(mc_info, cut_name=cuts)
            mc_info = mc_info[evt_sel_mask]
            print('Event number after selection: ' + str(mc_info.shape))
            for key in dset_keys:
                preds.append(pred_file[key][evt_sel_mask])

    else:
        for key in dset_keys:
            preds.append(pred_file[key])

    return (mc_info, ) + tuple(preds)


# --------------------------- Code for making cuts --------------------------- #
