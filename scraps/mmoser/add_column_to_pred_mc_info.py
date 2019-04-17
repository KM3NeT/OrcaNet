#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for making performance plots based on nn model predictions.
"""

import numpy as np
import h5py
import numpy.lib.recfunctions
import os
import shutil


def view_1d(a, b):  # a, b are arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(),  b.view(void_dt).ravel()


def argwhere_nd_searchsorted(a,b):
    A, B = view_1d(a, b)
    sidxB = B.argsort()
    mask = np.isin(A, B)
    cm = A[mask]
    idx0 = np.flatnonzero(mask)
    idx1 = sidxB[np.searchsorted(B,cm, sorter=sidxB)]
    return idx0, idx1  # idx0 : indices in A, idx1 : indices in B


def copy_and_rename_file(path_file):

    dirpath = os.path.dirname(path_file)
    fname = os.path.basename(path_file)
    fname_wout_ext = os.path.splitext(fname)[0]

    path_file_copy = dirpath + '/' + fname_wout_ext + '_w_weight_col.h5'

    shutil.copy(path_file, path_file_copy)

    return path_file_copy


path_std_reco_file = '/home/saturn/capn/mppi033h/Data/standard_reco_files/new_04_18/pred_file_bg_classifier_2_class.h5'
path_pred_file_dl = '/home/woody/capn/mppi033h/orcanet_trainings/bg_classifier/2_class_bg/predictions/concatenated/pred_model_epoch_2_file_2_on_bg_list_full_val_incl_rest_all_val_files.h5'

path_pred_file_dl_copy = copy_and_rename_file(path_pred_file_dl)

f_std = h5py.File(path_std_reco_file, 'r+')
f_dl = h5py.File(path_pred_file_dl_copy, 'r+')

mc_info_dl, mc_info_std = f_dl['mc_info'][()], f_std['mc_info'][()]

ax = np.newaxis
mc_info_dl_id = np.concatenate([mc_info_dl['run_id'][:, ax], mc_info_dl['event_id'][:, ax],
                                mc_info_dl['prod_ident'][:, ax], mc_info_dl['particle_type'][:, ax],
                                mc_info_dl['is_cc'][:, ax]], axis=1)
mc_info_std_id = np.concatenate([mc_info_std['run_id'][:, ax], mc_info_std['event_id'][:, ax],
                                 mc_info_std['prod_ident'][:, ax], mc_info_std['particle_type'][:, ax],
                                 mc_info_std['is_cc'][:, ax]], axis=1)

# append weight column to dl file, initialize with nans
print('Appending a weight column to the mc_info of the dl pred file with dummy nan values.')
w_dummy_vals = np.full(mc_info_dl.shape[0], np.nan)
mc_info_dl = np.lib.recfunctions.append_fields(mc_info_dl, 'oscillated_weight_one_year_bg_sel', w_dummy_vals, dtypes=[np.float64], usemask=False)

print('Searching for where columns of the dl pred file are located in the std reco file.')
idx_dl, idx_std = argwhere_nd_searchsorted(mc_info_dl_id, mc_info_std_id)

print(str(idx_dl.shape[0]) + ' events of the dl file are found in the std reco file.')

# Fix osc_w_1_y, since we dont use the full statistics in the contamination plot later!!
osc_w_1_y = mc_info_std['oscillated_weight_one_year'][()]
ptype_std, is_cc_std = mc_info_std['particle_type'], mc_info_std['is_cc']

# get boolean masks for different ptypes and sim prods
is_mupage = np.abs(ptype_std) == 13
is_nu_e_cc = np.logical_and(np.abs(ptype_std) == 12, is_cc_std == 1)
is_nu_e_nc = np.logical_and(np.abs(ptype_std) == 12, is_cc_std == 0)
is_nu_mu_cc = np.logical_and(np.abs(ptype_std) == 14, is_cc_std == 1)
is_1_5 = mc_info_std['prod_ident'] == 2
is_3_100 = mc_info_std['prod_ident'] == 1

# define correction values, inverse of n_runs_in_val_set / n_runs_total
w_corr_val_e_cc_1_5, w_corr_val_e_cc_3_100 = 600/180, 600/360
w_corr_val_e_nc_1_5, w_corr_val_e_nc_3_100 = 600/180, 600/360
w_corr_val_mu_cc_1_5, w_corr_val_mu_cc_3_100 = 600/180, 1
w_corr_val_mupage = 20000/15617

# apply correction
osc_w_1_y[np.logical_and(is_nu_e_cc, is_1_5)] = osc_w_1_y[np.logical_and(is_nu_e_cc, is_1_5)] * w_corr_val_e_cc_1_5
osc_w_1_y[np.logical_and(is_nu_e_cc, is_3_100)] = osc_w_1_y[np.logical_and(is_nu_e_cc, is_3_100)] * w_corr_val_e_cc_3_100

osc_w_1_y[np.logical_and(is_nu_e_nc, is_1_5)] = osc_w_1_y[np.logical_and(is_nu_e_nc, is_1_5)] * w_corr_val_e_nc_1_5
osc_w_1_y[np.logical_and(is_nu_e_nc, is_3_100)] = osc_w_1_y[np.logical_and(is_nu_e_nc, is_3_100)] * w_corr_val_e_nc_3_100

osc_w_1_y[np.logical_and(is_nu_mu_cc, is_1_5)] = osc_w_1_y[np.logical_and(is_nu_mu_cc, is_1_5)] * w_corr_val_mu_cc_1_5
osc_w_1_y[np.logical_and(is_nu_mu_cc, is_3_100)] = osc_w_1_y[np.logical_and(is_nu_mu_cc, is_3_100)] * w_corr_val_mu_cc_3_100

osc_w_1_y[is_mupage] = osc_w_1_y[is_mupage] * w_corr_val_mupage

# Finished fixing, now fill the rows of the dl pred_file with the w_1_y vals
print('Filling the rows of the weight col of the dl_pred file with the appropriate weight values from the std reco.')
mc_info_dl['oscillated_weight_one_year_bg_sel'][idx_dl] = osc_w_1_y[idx_std]

del f_dl['mc_info']
f_dl.create_dataset('mc_info', data=mc_info_dl, chunks=(64,), compression='gzip', compression_opts=1)
f_dl.close()

# Also fix it in the standard reco file!!
mc_info_std = np.lib.recfunctions.append_fields(mc_info_std, 'oscillated_weight_one_year_bg_sel', osc_w_1_y, dtypes=[np.float64], usemask=False)

del f_std['mc_info']
f_std.create_dataset('mc_info', data=mc_info_std, chunks=(64,), compression='gzip', compression_opts=1)
f_std.close()





