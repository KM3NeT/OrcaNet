#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""""

import numpy as np



def cut_summary_file():
    col_names = {}
    fpath_col_names = '/home/woody/capn/mppi033h/Data/various/pid_result_shiftedVertexEventSelection_withDeepLearningTrackScore_column_names.txt'

    with open(fpath_col_names) as f:
        f_cols = f.readlines()

    for i, col in enumerate(f_cols):
        col_names[col.strip('\n')] = i

    usecols = (col_names['bjorkeny'], col_names['dir_x'], col_names['dir_y'], col_names['dir_z'], col_names['energy'],
               col_names['event_id'], col_names['is_cc'], col_names['is_neutrino'], col_names['n_files_gen'], col_names['run_id'], col_names['type'],
               col_names['dusj_best_DusjOrcaUsingProbabilitiesFinalFit_BjorkenY'], col_names['dusj_dir_x'],
               col_names['dusj_dir_y'], col_names['dusj_dir_z'], col_names['dusj_is_good'],
               col_names['dusj_energy_corrected'], col_names['gandalf_dir_x'], col_names['gandalf_dir_y'],
               col_names['gandalf_dir_z'], col_names['gandalf_is_good'], col_names['gandalf_energy_corrected'], col_names['dusj_is_selected'], col_names['gandalf_is_selected'])

    summary_file_arr = np.genfromtxt('/home/woody/capn/mppi033h/Data/various/pid_result_shiftedVertexEventSelection_withDeepLearningTrackScore.meta', delimiter=' ', usecols=usecols)
    boolean_is_neutrino = summary_file_arr[:,7] == True
    summary_file_arr_neutr = summary_file_arr[boolean_is_neutrino].astype(np.float32)

    np.save('/home/woody/capn/mppi033h/Data/various/summary_file_cut.npy', summary_file_arr_neutr)


def make_arr_nn_pred():

    sum_arr = np.load('/home/woody/capn/mppi033h/Data/various/summary_file_cut.npy')
    ptype_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1),
                           'a_elec-CC': (-12, 1), 'elec-NC': (12, 0), 'a_elec-NC': (-12, 0),
                           'tau-CC': (16, 1), 'a_tau-CC': (-16, 1)}

    arr_nn_pred = np.zeros((sum_arr.shape[0], 19), dtype=np.float32)

    n_total_events, n_events_muon_cc, n_events_muon_cc_sel, n_events_elec_cc, n_events_elec_cc_sel = 0, 0, 0, 0, 0
    for i in xrange(sum_arr.shape[0]):
        if i % 100000 == 0: print 'Calculating the arr_nn_pred in step ', i

        n_files_gen = sum_arr[i, 8]

        if n_files_gen not in [550, 598]: # select 3-100GeV prod
            continue

        run_id = sum_arr[i, 9]
        event_id = sum_arr[i, 5]
        particle_type = sum_arr[i, 10]
        is_cc = sum_arr[i, 6]
        energy_mc = sum_arr[i, 4]
        bjorken_y_mc = sum_arr[i, 0]
        dir_x_mc, dir_y_mc, dir_z_mc = sum_arr[i, 1], sum_arr[i, 2], sum_arr[i, 3]

        if energy_mc < 3: print energy_mc

        dusj_is_good = sum_arr[i, 15]
        gandalf_is_good = sum_arr[i, 20]
        dusj_is_selected = sum_arr[i, 22]
        gandalf_is_selected = sum_arr[i, 23]

        n_total_events += 1

        if (particle_type, is_cc) == ptype_dict['elec-CC'] or (particle_type, is_cc) == ptype_dict['a_elec-CC']: # shower
            n_events_elec_cc += 1
            #if dusj_is_good:
            if dusj_is_selected:
                n_events_elec_cc_sel += 1
                energy_pred = sum_arr[i, 16]
                dir_x_pred, dir_y_pred, dir_z_pred = sum_arr[i, 12], sum_arr[i, 13], sum_arr[i, 14]
                bjorken_y_pred = sum_arr[i, 11]
            else:
                continue

        elif (particle_type, is_cc) == ptype_dict['muon-CC'] or (particle_type, is_cc) == ptype_dict['a_muon-CC']: # track
            n_events_muon_cc += 1
            #if gandalf_is_good:
            if gandalf_is_selected:
                n_events_muon_cc_sel += 1
                energy_pred = sum_arr[i, 21]
                dir_x_pred, dir_y_pred, dir_z_pred = sum_arr[i, 17], sum_arr[i, 18], sum_arr[i, 19]
                bjorken_y_pred = np.array(0, dtype=np.float32) # not available for gandalf
            else:
                continue

        else:
            continue

        mc_info = np.array([run_id, event_id, particle_type, is_cc, energy_mc, bjorken_y_mc, dir_x_mc, dir_y_mc, dir_z_mc], dtype=np.float32)
        y_pred = np.array([energy_pred, dir_x_pred, dir_y_pred, dir_z_pred, bjorken_y_pred], dtype=np.float32)
        y_true = np.array([energy_mc, dir_x_mc, dir_y_mc, dir_z_mc, bjorken_y_mc], dtype=np.float32)

        arr_nn_pred[i, :] = np.concatenate([mc_info, y_pred, y_true], axis=0)

    print 'Total number of events in the summary file: ', n_total_events
    print 'Number of elec-CC events: ', n_events_elec_cc
    print 'Number of selected elec-CC events: ', n_events_elec_cc_sel
    print 'You have thrown away ' + str(np.round((1- n_events_elec_cc_sel/float(n_events_elec_cc)) * 100, 1)) + '% of the events'
    print 'Number of muon-CC events: ', n_events_muon_cc
    print 'Number of selected muon-CC events: ', n_events_muon_cc_sel
    print 'You have thrown away ' + str(np.round((1- n_events_muon_cc_sel/float(n_events_muon_cc)) * 100, 1)) + '% of the events'

    # remove lines with only 0 entries
    arr_nn_pred = arr_nn_pred[~(arr_nn_pred==0).all(1)]
    np.save('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy', arr_nn_pred)

    # save cut info (run_id, evt_id) to txt
    boolean_muon_cc = np.abs(arr_nn_pred[:, 2:4]) == np.array([14, 1])
    boolean_elec_cc = np.abs(arr_nn_pred[:, 2:4]) == np.array([12, 1])
    indices_rows_with_muon_cc = np.logical_and(boolean_muon_cc[:, 0], boolean_muon_cc[:, 1])
    indices_rows_with_elec_cc = np.logical_and(boolean_elec_cc[:, 0], boolean_elec_cc[:, 1])
    arr_nn_pred_muon_cc = arr_nn_pred[indices_rows_with_muon_cc ]
    arr_nn_pred_elec_cc = arr_nn_pred[indices_rows_with_elec_cc ]

    run_and_evt_id_muon_cc = arr_nn_pred_muon_cc[:, 0:2].astype('int')
    run_and_evt_id_elec_cc = arr_nn_pred_elec_cc[:, 0:2].astype('int')
    np.savetxt('/home/woody/capn/mppi033h/Data/various/cuts_shallow_3_100_muon_cc.txt', run_and_evt_id_muon_cc, delimiter=' ', fmt="%d", header='run_id event_id', newline='\n')
    np.savetxt('/home/woody/capn/mppi033h/Data/various/cuts_shallow_3_100_elec_cc.txt', run_and_evt_id_elec_cc, delimiter=' ', fmt="%d", header='run_id event_id', newline='\n')


def make_shallow_energy_plots():
    from utilities.evaluation_utilities import make_2d_energy_resolution_plot, make_1d_energy_reco_metric_vs_energy_plot

    arr_nn_pred = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')

    make_2d_energy_resolution_plot(arr_nn_pred, 'shallow_reco', compare_pheid=(True, '3-100_GeV_prod_energy_comparison'))
    make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, 'shallow_reco', metric='median', energy_bins=np.linspace(3, 100, 32), compare_pheid=(True, '3-100_GeV_prod_energy_comparison'))


if __name__ == '__main__':
    #cut_summary_file() # do on tinyfat or woody with 32g
    make_arr_nn_pred()
    #make_shallow_energy_plots()






