#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""""

import numpy as np



def cut_summary_file():
    """

    """
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
               col_names['gandalf_dir_z'], col_names['gandalf_is_good'], col_names['gandalf_energy_corrected'], col_names['dusj_is_selected'],
               col_names['gandalf_is_selected'], col_names['frame_index'], col_names['track_score'], col_names['Erange_min'])

    summary_file_arr = np.genfromtxt('/home/woody/capn/mppi033h/Data/various/pid_result_shiftedVertexEventSelection_withDeepLearningTrackScore.meta', delimiter=' ', usecols=usecols)
    boolean_is_neutrino = summary_file_arr[:,7] == True
    summary_file_arr_neutr = summary_file_arr[boolean_is_neutrino].astype(np.float32)

    np.save('/home/woody/capn/mppi033h/Data/various/summary_file_cut.npy', summary_file_arr_neutr)


def make_arr_nn_pred():
    """

    """
    sum_arr = np.load('/home/woody/capn/mppi033h/Data/various/summary_file_cut.npy')

    ic_dict_str = {(14, 1): 'muon-CC', (12, 1): 'elec-CC', (12, 0): 'elec-NC', (16, 1): 'tau-CC'} # interaction channels
    # tuple[0]: all events ; tuple[1]: selected events
    n_events = {'total': [0, 0],'muon-CC': [0, 0], 'elec-CC': [0, 0], 'elec-NC': [0, 0], 'tau-CC': [0, 0]}

    arr_nn_pred = np.zeros((sum_arr.shape[0], 19), dtype=np.float32)

    for i in range(sum_arr.shape[0]):
        if i % 100000 == 0: print('Calculating the arr_nn_pred in step ', i)

        e_range_min = sum_arr[i, 26]
        energy_mc = sum_arr[i, 4]

        if e_range_min == 1 and energy_mc > 3: # throw away 3-5 GeV events from the low e 1-5 GeV prod
            continue

        run_id = sum_arr[i, 9]
        event_id = sum_arr[i, 24] # index 5: old event_id, index 24: new event_id, counts all generated mc events
        particle_type = sum_arr[i, 10]
        is_cc = sum_arr[i, 6]
        bjorken_y_mc = sum_arr[i, 0]
        dir_x_mc, dir_y_mc, dir_z_mc = sum_arr[i, 1], sum_arr[i, 2], sum_arr[i, 3]

        dusj_is_good = sum_arr[i, 15]
        gandalf_is_good = sum_arr[i, 20]
        dusj_is_selected = sum_arr[i, 22]
        gandalf_is_selected = sum_arr[i, 23]

        n_events['total'][0] += 1

        ic_str = ic_dict_str[(np.abs(particle_type), is_cc)] # just for convenience

        # separate branches for tracks and showers
        if ic_str in ['elec-CC', 'elec-NC', 'tau-CC']: # showers, TODO split taus
            n_events[ic_str][0] += 1

            #if dusj_is_good == 1:
            if dusj_is_selected == 1:
                n_events['total'][1] += 1
                n_events[ic_str][1] += 1

                energy_pred = sum_arr[i, 16]
                dir_x_pred, dir_y_pred, dir_z_pred = sum_arr[i, 12], sum_arr[i, 13], sum_arr[i, 14]
                bjorken_y_pred = sum_arr[i, 11]

            else:
                continue

        elif ic_str in ['muon-CC']:
            n_events[ic_str][0] += 1

            if gandalf_is_selected == 1:
                n_events['total'][1] += 1
                n_events[ic_str][1] += 1

                energy_pred = sum_arr[i, 21]
                dir_x_pred, dir_y_pred, dir_z_pred = sum_arr[i, 17], sum_arr[i, 18], sum_arr[i, 19]
                bjorken_y_pred = np.array(0.25715, dtype=np.float32) # not available for gandalf, use median of by_true

            else:
                continue

        else:
            raise ValueError('This line of code will never be reached.')

        mc_info = np.array([run_id, event_id, particle_type, is_cc, energy_mc, bjorken_y_mc, dir_x_mc, dir_y_mc, dir_z_mc], dtype=np.float32)
        y_pred = np.array([energy_pred, dir_x_pred, dir_y_pred, dir_z_pred, bjorken_y_pred], dtype=np.float32)
        y_true = np.array([energy_mc, dir_x_mc, dir_y_mc, dir_z_mc, bjorken_y_mc], dtype=np.float32)

        arr_nn_pred[i, :] = np.concatenate([mc_info, y_pred, y_true], axis=0)


    for ic in ['total', 'muon-CC', 'elec-CC', 'elec-NC', 'tau-CC']:
        print('Number of ' + ic +' events in the summary file: ', n_events[ic][0])
        print('Number of selected ' + ic + ' events in the summary file: ', n_events[ic][1])
        print('You have thrown away ' + str(np.round((1 - n_events[ic][1] / float(n_events[ic][0])) * 100, 1)) + '% of the events')

    # remove lines with only 0 entries
    arr_nn_pred = arr_nn_pred[~(arr_nn_pred==0).all(1)]
    np.save('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy', arr_nn_pred)

    # save cut info (run_id, evt_id) to txt
    for n_list, ic in ic_dict_str.items():
        boolean_ic = np.abs(arr_nn_pred[:, 2:4]) == np.array(n_list)
        indices_rows_with_ic = np.logical_and(boolean_ic[:, 0], boolean_ic[:, 1])
        arr_nn_pred_ic = arr_nn_pred[indices_rows_with_ic]

        # split to low_e and high_e prod, for low_e prod (1-5 GeV), only events from 1-3 GeV have been added to the arr_nn_pred!
        arr_nn_pred_ic_low_e = arr_nn_pred_ic[arr_nn_pred_ic[:, 4] < 3]
        arr_nn_pred_ic_high_e = arr_nn_pred_ic[arr_nn_pred_ic[:, 4] > 3]

        run_and_evt_id_ic_low_e = arr_nn_pred_ic_low_e[:, 0:2].astype('int')
        run_and_evt_id_ic_high_e = arr_nn_pred_ic_high_e[:, 0:2].astype('int')

        np.savetxt('/home/woody/capn/mppi033h/Data/various/cuts_txt_files/cuts_shallow_1_3_' + ic.lower() + '.txt',
                   run_and_evt_id_ic_low_e, delimiter=' ', fmt="%d", header='run_id event_id', newline='\n')
        np.savetxt('/home/woody/capn/mppi033h/Data/various/cuts_txt_files/cuts_shallow_3_100_' + ic.lower() + '.txt',
                   run_and_evt_id_ic_high_e, delimiter=' ', fmt="%d", header='run_id event_id', newline='\n')


if __name__ == '__main__':
    #cut_summary_file() # do on tinyfat or woody with 32g, qsub -I -l nodes=1:ppn=4:sl32g,walltime=05:00:00
    make_arr_nn_pred()






