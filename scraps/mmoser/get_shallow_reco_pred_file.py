#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""""

import os
import numpy as np
import numpy.lib.recfunctions
import h5py
from orcanet_contrib.plotting.bg_classifier import select_class
from orcanet_contrib.plotting.utils import select_track_shower


def cut_summary_file():
    """

    """
    col_names = {}
    # fpath_col_names = '/home/saturn/capn/mppi033h/Data/standard_reco_files/pid_result_shiftedVertexEventSelection_withDeepLearningTrackScore_column_names.txt'
    # meta_filepath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/pid_result_shiftedVertexEventSelection_withDeepLearningTrackScore.meta'
    # savepath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/summary_file_cut.h5'

    fpath_col_names = '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pid_result_shiftedVertexEventSelection_column_names.txt'
    meta_filepath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pid_result_shiftedVertexEventSelection_w_osc_w_1_y.meta'
    savepath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/summary_file_cut.h5'

    with open(fpath_col_names) as f_col_names:
        cols_list = f_col_names.read().splitlines()

    for i, col in enumerate(cols_list):
        col_names[col] = i

    selected_cols = ('bjorkeny', 'dir_x', 'dir_y', 'dir_z', 'energy', 'event_id', 'frame_index', 'is_cc', 'is_neutrino', 'n_files_gen',
                     'pos_x', 'pos_y', 'pos_z', 'run_id', 'time', 'trigger_counter', 'type', 'Erange_min', 'dusj_best_DusjOrcaUsingProbabilitiesFinalFit_BjorkenY',
                     'dusj_dir_x', 'dusj_dir_y', 'dusj_dir_z', 'dusj_pos_x', 'dusj_pos_y', 'dusj_pos_z', 'dusj_time',
                     'dusj_is_good', 'dusj_energy_corrected', 'gandalf_dir_x',
                     'gandalf_dir_y', 'gandalf_dir_z', 'gandalf_pos_x', 'gandalf_pos_y', 'gandalf_pos_z', 'gandalf_time',
                     'gandalf_is_good', 'gandalf_energy_corrected', 'dusj_is_selected', 'gandalf_is_selected',
                     'gandalf_loose_is_selected', 'muon_score', 'track_score', 'noise_score', 'oscillated_weight_one_year')

    usecols = [col_names[col_name] for col_name in selected_cols]
    usecols.sort()

    print('Loading the summary file content')
    dtype = dict()
    dtype['names'] = selected_cols
    dtype['formats'] = (np.float64, ) * len(selected_cols)
    summary_file_arr = np.loadtxt(meta_filepath, delimiter=' ', usecols=usecols, dtype=dtype)

    # # add weight_1_year_col
    # w = np.load('/home/saturn/capn/mppi033h/Data/standard_reco_files/weight_1_year_col.npy')
    # summary_file_arr = np.lib.recfunctions.append_fields(summary_file_arr, 'oscillated_weight_one_year', w, dtypes=[np.float64], usemask=False)

    f = h5py.File(savepath, 'w')
    print('Saving the summary file content')
    f.create_dataset('summary_array', data=summary_file_arr, compression='gzip', compression_opts=1)
    f.close()


def make_pred_file():
    """

    """
    f_in = h5py.File('/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/summary_file_cut.h5', 'r')

    # bg classifier selections
    print('Making files for bg classifier')
    s = f_in['summary_array']
    dusj_is_selected = s['dusj_is_selected']
    gandalf_loose_is_selected = s['gandalf_loose_is_selected']
    selection_classifier_bg = np.logical_or(dusj_is_selected, gandalf_loose_is_selected)

    mc_info_sel_bg = get_mc_info(s, selection=selection_classifier_bg)
    save_surviving_evt_info_to_npy(mc_info_sel_bg, '/home/saturn/capn/mppi033h/Data/event_selections/evt_selection_bg_classifier.npy')

    pred_bg = get_pred(s, 'bg_classifier_2_class', selection=selection_classifier_bg)
    save_dsets_to_h5_file(mc_info_sel_bg, pred_bg,
                          '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_bg_classifier_2_class.h5')

    # ts classifier
    print('Making files for ts classifier')
    is_neutrino = f_in['summary_array']['is_neutrino'] == 1

    s = f_in['summary_array'][is_neutrino]
    dusj_is_selected = s['dusj_is_selected']
    gandalf_loose_is_selected = s['gandalf_loose_is_selected']
    selection_classifier_ts = np.logical_or(dusj_is_selected, gandalf_loose_is_selected)

    mc_info_sel_ts = get_mc_info(s, selection=selection_classifier_ts)
    save_surviving_evt_info_to_npy(mc_info_sel_ts, '/home/saturn/capn/mppi033h/Data/event_selections/evt_selection_ts_classifier.npy')

    pred_ts = get_pred(s, 'ts_classifier', selection=selection_classifier_ts)
    save_dsets_to_h5_file(mc_info_sel_ts, pred_ts,
                          '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_ts_classifier.h5')

    # regression
    print('Making files for regression')
    is_neutrino = f_in['summary_array']['is_neutrino'] == 1
    s = f_in['summary_array'][is_neutrino]
    dusj_is_selected = s['dusj_is_selected']
    gandalf_is_selected = s['gandalf_is_selected']
    selection_regr = np.logical_or(dusj_is_selected, gandalf_is_selected)

    mc_info_sel_regr = get_mc_info(s, selection=selection_regr)
    save_surviving_evt_info_to_npy(mc_info_sel_regr, '/home/saturn/capn/mppi033h/Data/event_selections/evt_selection_regression.npy')

    pred_regr = get_pred(s, 'regression_e_dir_vtx_by', selection=selection_regr)
    save_dsets_to_h5_file(mc_info_sel_regr, pred_regr,
                          '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_regression.h5')

    f_in.close()

    # TODO make print how many events have been thrown away


def get_mc_info(s, selection=None):
    """

    Parameters
    ----------
    s
    selection

    Returns
    -------
    mc_info

    """
    if selection is not None:
        s = s[selection]

    cols_s_out = [('type', 'particle_type'), ('energy', 'energy'),
                  ('is_cc', 'is_cc'), ('bjorkeny', 'bjorkeny'), ('dir_x', 'dir_x'),
                  ('dir_y', 'dir_y'), ('dir_z', 'dir_z'), ('time', 'time_interaction'),
                  ('run_id', 'run_id'), ('pos_x', 'vertex_pos_x'), ('pos_y', 'vertex_pos_y'),
                  ('pos_z', 'vertex_pos_z'), ('oscillated_weight_one_year', 'oscillated_weight_one_year')]

    dtypes = [(tpl[1], s[tpl[0]].dtype) for tpl in cols_s_out]

    # make prod_ident col
    prod_ident = np.zeros(s['run_id'].shape[0], dtype=np.float64)

    is_mupage = select_class('mupage', s['type'], s['is_cc'])
    is_random_noise = select_class('random_noise', s['type'], s['is_cc'])
    is_neutrino = np.invert(np.logical_or(is_mupage, is_random_noise))
    is_neutrino_low_e = np.logical_and(s['Erange_min'] == 1, is_neutrino)
    is_neutrino_high_e = np.logical_and(np.invert(is_neutrino_low_e), is_neutrino)

    assert np.count_nonzero(is_neutrino) == np.count_nonzero(is_neutrino_low_e) + np.count_nonzero(is_neutrino_high_e)

    prod_ident[is_mupage] = 3
    prod_ident[is_random_noise] = 4
    prod_ident[is_neutrino_low_e] = 2
    prod_ident[is_neutrino_high_e] = 1
    assert np.count_nonzero(prod_ident == 0) == 0

    cols_s_out.append((None, 'prod_ident'))
    dtypes.append(('prod_ident', np.float64))

    # for random noise files, we need to take the trigger_counter as event_id, not the frame_index!
    event_id = np.zeros(s['run_id'].shape[0], dtype=np.float64)
    event_id[is_random_noise] = s['trigger_counter'][is_random_noise]
    event_id[np.invert(is_random_noise)] = s['frame_index'][np.invert(is_random_noise)]
    cols_s_out.append((None, 'event_id'))
    dtypes.append(('event_id', np.float64))

    n_evts = s['energy'].shape[0]
    mc_info = np.empty(n_evts, dtype=dtypes)
    for tpl in cols_s_out:
        if tpl[1] == 'prod_ident':
            mc_info['prod_ident'] = prod_ident
        elif tpl[1] == 'event_id':
            mc_info['event_id'] = event_id
        else:
            mc_info[tpl[1]] = s[tpl[0]]

    return mc_info


def save_surviving_evt_info_to_npy(mc_info, savepath):
    """

    Parameters
    ----------
    mc_info
    savepath

    """
    ax = np.newaxis
    arr = np.concatenate([mc_info['run_id'][:, ax], mc_info['event_id'][:, ax],
                          mc_info['prod_ident'][:, ax], mc_info['particle_type'][:, ax],
                          mc_info['is_cc'][:, ax]], axis=1)

    np.save(savepath, arr)


def get_pred(s, mode, selection=None):
    """

    Parameters
    ----------
    s
    mode
    selection

    Returns
    -------
    pred

    """
    if selection is not None:
        s = s[selection]

    if mode == 'bg_classifier':
        raise ValueError('bg_classifier not yet implemented')
        # need 3 values, prob mupage, prob random_noise, prob neutrino

    elif mode == 'bg_classifier_2_class':
        # need 2 values, prob neutrino, prob not neutrino

        muon_score, noise_score = s['muon_score'], s['noise_score']

        dtypes_pred = [('prob_muon', np.float64), ('prob_noise', np.float64)]
        n_evts = muon_score.shape[0]
        pred = np.empty(n_evts, dtype=dtypes_pred)

        pred['prob_muon'] = muon_score
        pred['prob_noise'] = noise_score

        # prob_not_neutrino = s['noise_score']
        # prob_neutrino = 1 - prob_not_neutrino
        #
        # dtypes_pred = [('prob_not_neutrino', np.float64), ('prob_neutrino', np.float64)]
        # n_evts = prob_not_neutrino.shape[0]
        # pred = np.empty(n_evts, dtype=dtypes_pred)
        #
        # pred['prob_neutrino'] = prob_neutrino
        # pred['prob_not_neutrino'] = prob_not_neutrino

    elif mode == 'ts_classifier':
        # need 2 values, prob shower, prob track

        prob_track = s['track_score']
        prob_shower = 1 - prob_track

        dtypes_pred = [('prob_track', np.float64), ('prob_shower', np.float64)]
        n_evts = prob_track.shape[0]
        pred = np.empty(n_evts, dtype=dtypes_pred)

        pred['prob_track'] = prob_track
        pred['prob_shower'] = prob_shower

    elif mode == 'regression_e_dir_vtx_by':
        is_track, is_shower = select_track_shower(s['type'], s['is_cc'])
        s_shower, s_track = s[is_shower], s[is_track]

        # energy
        energy = np.empty(s.shape[0], dtype=np.float64)
        energy[is_shower], energy[is_track] = s_shower['dusj_energy_corrected'], s_track['gandalf_energy_corrected']
        assert np.all(np.logical_or(is_shower, is_track)) == True

        # bjorkeny
        bjorkeny = np.empty(s.shape[0], dtype=np.float64)
        bjorkeny[is_shower] = s_shower['dusj_best_DusjOrcaUsingProbabilitiesFinalFit_BjorkenY']
        bjorkeny[is_track] = np.full(s_track.shape, 0.360366)  # not available for gandalf, use median of by_true of training dataset

        # dir and vtx
        labels_show_track = (('dusj_dir_x', 'gandalf_dir_x'), ('dusj_dir_y', 'gandalf_dir_y'), ('dusj_dir_z', 'gandalf_dir_z'),
                             ('dusj_pos_x', 'gandalf_pos_x'), ('dusj_pos_y', 'gandalf_pos_y'), ('dusj_pos_z', 'gandalf_pos_z'),
                             ('dusj_time', 'gandalf_time'))
        dir_vtx = {}
        for i, label_tpl in enumerate(labels_show_track):
            dir_vtx[i] = np.empty(s.shape[0], dtype=np.float64)
            dir_vtx[i][is_shower] = s_shower[label_tpl[0]]
            dir_vtx[i][is_track] = s_track[label_tpl[1]]

        # save everything to a new pred dset
        labels_pred_dset = ['pred_energy', 'pred_dir_x', 'pred_dir_y', 'pred_dir_z', 'pred_bjorkeny', 'pred_vtx_x',
                            'pred_vtx_y', 'pred_vtx_z', 'pred_vtx_t']

        dtypes_pred = [(label_pred_name, np.float64) for label_pred_name in labels_pred_dset]
        n_evts = energy.shape[0]
        pred = np.empty(n_evts, dtype=dtypes_pred)

        pred['pred_energy'] = energy
        pred['pred_bjorkeny'] = bjorkeny
        pred['pred_dir_x'], pred['pred_dir_y'], pred['pred_dir_z'] = dir_vtx[0], dir_vtx[1], dir_vtx[2]
        pred['pred_vtx_x'], pred['pred_vtx_y'], pred['pred_vtx_z'], pred['pred_vtx_t'] = dir_vtx[3], dir_vtx[4], dir_vtx[5], dir_vtx[6]

    return pred


def save_dsets_to_h5_file(mc_info, pred, savepath):
    """

    Parameters
    ----------
    mc_info
    pred
    savepath

    """
    f = h5py.File(savepath, 'w')
    f.create_dataset('mc_info', data=mc_info, compression='gzip', compression_opts=1)
    f.create_dataset('pred', data=pred, compression='gzip', compression_opts=1)
    f.close()


if __name__ == '__main__':
    if os.path.exists('/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/summary_file_cut.h5') is False:
        cut_summary_file()  # do on tinyfat or woody with 32g, qsub -I -l nodes=1:ppn=4:sl32g,walltime=05:00:00

    make_pred_file()






