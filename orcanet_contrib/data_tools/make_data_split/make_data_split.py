#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

TODO

Usage:
    make_data_split.py CONFIG
    make_data_split.py (-h | --help)

Arguments:
    CONFIG  A .toml file which sets up TODO

Options:
    -h --help  Show this screen.

"""

import os
import toml
import docopt
import natsort as ns
import h5py

def parse_input():

    args = docopt.docopt(__doc__)
    config_file = args['CONFIG']

    cfg = toml.load(config_file)

    return cfg


def get_all_ip_group_keys(cfg):
    """

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    ip_group_keys = []
    for key in cfg:
        if type(key) == dict:
            ip_group_keys.append(key)

    return ip_group_keys


def get_h5_filepaths(dirpath):
    """
    Returns the filepaths of all .h5 files that are located in a specific directory.
    :param str dirpath: path of the directory where the .h5 files are located.
    :return: list filepaths: list with the full filepaths of all .h5 files in the dirpath folder.
    """
    filepaths = []
    for f in os.listdir(dirpath):
        if f.endswith(".h5"):
            filepaths.append(f)

    filepaths = ns.natsorted(filepaths)
    return filepaths


def get_number_of_evts_and_run_ids(list_of_files, dataset_key='y', run_id_col_name='run_id'):

    total_number_of_evts = 0
    run_id_list = []

    for fpath in list_of_files:
        f = h5py.File(fpath, 'r')

        dset = f[dataset_key]
        n_evts = dset.shape[0]
        total_number_of_evts += n_evts

        run_id = f[dataset_key][0][run_id_col_name]
        run_id_list.append(run_id)

        f.close()

    mean_number_of_evts_per_file = total_number_of_evts / len(list_of_files)

    return total_number_of_evts, mean_number_of_evts_per_file, run_id_list


def split(a, n):
    # from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_run_ranges(cfg, ip_group_keys, mode='train'):

    # If the input groups do not have the needed number of events per class for the specified contribution, we
    # need to move some leftover events to a leftover data split.
    # In order to calculate this appropriately, we need to get the group, where the difference between the needed
    # number of events for the (train) datasplit and the actually available number of events is the highest.

    group_key_max_diff, diff_max_to_avail = None, 0
    for key in ip_group_keys:
        n_evts_train_max = int(cfg['n_evts_total'] * cfg['fraction_train'] * cfg[key]['contribution_train'])
        n_evts_train_avail = int(cfg[key]['n_events'] * cfg[key]['fraction_train'])
        cfg[key]['n_evts_train_max'] = n_evts_train_max

        diff_max_avail_key = n_evts_train_max - n_evts_train_avail
        # TODO what happens if every group has a diff of 0, or if it is negative?
        if group_key_max_diff is None or diff_max_avail_key > diff_max_to_avail:
            group_key_max_diff = key

    # train files should start from highest run_id

    # run_id_min_max = (min(cfg[group_key_max_diff]['run_id_list']), max(cfg[group_key_max_diff]['run_id_list']))

    n_evts_train = cfg[group_key_max_diff]['n_evts'] * cfg['fraction_train']
    n_files_train = int(n_evts_train / cfg[group_key_max_diff]['n_evts_per_file_mean'])

    #run_id_end = run_id_min_max[1]
    file_start_index = len(cfg[group_key_max_diff]['fpaths']) - n_files_train
    file_start = cfg[group_key_max_diff]['fpaths'][file_start_index]
    file_end = cfg[group_key_max_diff]['fpaths'][-1]

    filelist_cut = cfg[group_key_max_diff]['fpaths'][file_start_index:-1]
    file_splits = split(filelist_cut, cfg['n_train_files'])



def make_data_split():
    cfg = parse_input()
    # TODO check if user input makes sense

    ip_group_keys = get_all_ip_group_keys(cfg)

    n_evts_total = 0
    for key in ip_group_keys:
        cfg[key]['fpaths'] = get_h5_filepaths(cfg[key]['dir'])
        cfg[key]['n_files'] = len(cfg[key]['fpaths'])
        cfg[key]['n_evts'], cfg[key]['n_evts_per_file_mean'], cfg[key]['run_id_list'] = get_number_of_evts_and_run_ids(cfg[key]['fpaths'], dataset_key='y')

        n_evts_total += cfg[key]['n_evts']

    cfg['n_evts_total'] = n_evts_total






if __name__ == '__main__':
    make_data_split()


