#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Michael's orcanet utility stuff.

"""
import numpy as np
import toml

from orcanet_contrib.custom_objects import get_custom_objects


def update_objects(orga, model_file):
    """
    Update the organizer for using the model.

    Look up and load in the respective sample-, label-, and dataset-
    modifiers, as well as the custom objects.
    Will assert that the respective objects have not already been set
    to a non-default value (nothing is overwritten).

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model_file : str
        Path to a toml file which has the infos about which modifiers
        to use.

    """
    file_content = toml.load(model_file)
    orca_modifiers = file_content["orca_modifiers"]

    sample_modifier = orca_modifiers.get("sample_modifier")
    label_modifier = orca_modifiers.get("label_modifier")
    dataset_modifier = orca_modifiers.get("dataset_modifier")

    if sample_modifier is not None:
        print("Using orga sample modifier: ", sample_modifier)
        orga.cfg.sample_modifier = orca_sample_modifiers(sample_modifier)
    if label_modifier is not None:
        print("Using orga label modifier: ", label_modifier)
        orga.cfg.label_modifier = orca_label_modifiers(label_modifier)
    if dataset_modifier is not None:
        print("Using orga dataset modifier: ", dataset_modifier)
        orga.cfg.dataset_modifier = orca_dataset_modifiers(dataset_modifier)
    print("Using orga custom objects")
    orga.cfg.custom_objects = get_custom_objects()


def orca_sample_modifiers(name):
    """
    Returns one of the sample modifiers used for Orca networks.

    They will permute columns, and/or add permuted columns to xs.

    The input to the functions is:
        xs_files : dict
            Dict that contains the input samples from the file(s).
            The keys are the names of the inputs in the toml list file.
            The values are a single batch of data from each corresponding file.

    The output is:
        xs_layer : dict
            Dict that contains the input samples for a Keras NN.
            The keys are the names of the input layers of the network.
            The values are a single batch of data for each input layer.

    Parameters
    ----------
    name : None/str
        Name of the sample modifier to return.

    Returns
    -------
    sample_modifier : function
        The sample modifier function.

    """
    # assuming input is bxyzt
    xyzt_permute = {'yzt-x': (0, 2, 3, 4, 1),
                    'xyt-z': (0, 1, 2, 4, 3),
                    't-xyz': (0, 4, 1, 2, 3),
                    'tyz-x': (0, 4, 2, 3, 1)}

    if name in xyzt_permute:
        def swap_columns(xs_files):
            # Transpose dimensions
            xs_layer = dict()
            keys = list(xs_files.keys())
            xs_layer[keys[0]] = np.transpose(xs_files[keys[0]], xyzt_permute[name])
            return xs_layer
        sample_modifier = swap_columns

    elif name == "sum_last":
        def sample_modifier(xs_files):
            # sum over the last dimension
            # e.g. shape (10,20,30) --> (10,20,1)
            xs_layer = dict()
            for l_name, x in xs_files.items():
                xs_layer[l_name] = np.sum(x, axis=-1, keepdims=True)
            return xs_layer

    elif name == 'xyz-t_and_yzt-x':
        def sample_modifier(xs_files):
            # Use xyz-t, and also transpose it to yzt-x and use that, too.
            xs_layer = dict()
            xs_layer['xyz-t'] = xs_files['xyz-t']
            xs_layer['yzt-x'] = np.transpose(xs_files['xyz-t'], xyzt_permute['yzt-x'])
            return xs_layer

    elif name == 'xyz-t_and_xyz-c_single_input_and_yzt-x':
        def sample_modifier(xs_files):
            # Concatenate xyz-t and xyz-c to a single input
            xs_layer = dict()
            xs_layer['xyz-t_and_xyz-c_single_input_net_0'] = np.concatenate(
                [xs_files['xyz-t'], xs_files['xyz-c']], axis=-1)
            # Transpose xyz-t to yzt-x and use that, too.
            xs_layer['input_1_net_1'] = np.transpose(xs_files['xyz-t'], xyzt_permute['yzt-x'])
            return xs_layer

    elif name == 'xyz-t_and_yzt-x_multi_input_single_train_tight-1_tight-2':
        def sample_modifier(xs_files):
            # Use xyz-t in two different time cuts, and also transpose them to yzt-x and use these, too.
            xs_layer = dict()
            xs_layer['xyz-t_tight-1'] = xs_files['xyz-t_tight-1']
            xs_layer['xyz-t_tight-2'] = xs_files['xyz-t_tight-2']
            xs_layer['yzt-x_tight-1'] = np.transpose(xs_files['xyz-t_tight-1'],
                                                     xyzt_permute['yzt-x'])
            xs_layer['yzt-x_tight-2'] = np.transpose(xs_files['xyz-t_tight-2'],
                                                     xyzt_permute['yzt-x'])
            return xs_layer

    elif name == 'xyz-t_and_xyz-c_single_input':
        def sample_modifier(xs_files):
            # Concatenate xyz-t and xyz-c to a single input
            xs_layer = dict()
            xs_layer['xyz-t_and_xyz-c_single_input'] = np.concatenate(
                [xs_files['xyz-t'], xs_files['xyz-c']], axis=-1)
            return xs_layer

    else:
        raise ValueError('Unknown input_type: ' + str(name))

    return sample_modifier


def orca_label_modifiers(name):
    """
    Returns one of the label modifiers used for Orca networks.

    CAREFUL: y_values is a structured numpy array! if you use advanced
    numpy indexing, this may lead to errors. Let's suppose you want to
    assign a particular value to one or multiple elements of the
    y_values array.

    E.g.
    y_values[1]['bjorkeny'] = 5
    This works, since it is basic indexing.

    Likewise,
    y_values[1:3]['bjorkeny'] = 5
    works as well, because basic indexing gives you a view (!).

    Advanced indexing though, gives you a copy.
    So this
    y_values[[1,2,4]]['bjorkeny'] = 5
    will NOT work! Same with boolean indexing, like

    bool_idx = np.array([True,False,False,True,False]) # if len(y_values) = 5
    y_values[bool_idx]['bjorkeny'] = 10
    This will NOT work as well!!

    Instead, use
    np.place(y_values['bjorkeny'], bool_idx, 10)
    This works.

    Parameters
    ----------
    name : str
        Name of the label modifier that should be used.

    Returns
    -------
    label_modifier : function
        The label modifier function.

    """

    if name == 'energy_dir_bjorken-y_vtx_errors':
        def label_modifier(y_values):
            ys = dict()
            particle_type, is_cc = y_values['particle_type'], y_values['is_cc']
            elec_nc_bool_idx = np.logical_and(np.abs(particle_type) == 12,
                                              is_cc == 0)

            # correct energy to visible energy
            visible_energy = y_values[elec_nc_bool_idx]['energy'] * y_values[elec_nc_bool_idx]['bjorkeny']
            # make a copy of the y_values array, since we modify it now
            y_values_copy = np.copy(y_values)
            # fix energy to visible energy
            np.place(y_values_copy['energy'], elec_nc_bool_idx, visible_energy)
            # set bjorkeny label of nc events to 1
            np.place(y_values_copy['bjorkeny'], elec_nc_bool_idx, 1)

            ys['dx'], ys['dx_err'] = y_values_copy['dir_x'], y_values_copy['dir_x']
            ys['dy'], ys['dy_err'] = y_values_copy['dir_y'], y_values_copy['dir_y']
            ys['dz'], ys['dz_err'] = y_values_copy['dir_z'], y_values_copy['dir_z']
            ys['e'], ys['e_err'] = y_values_copy['energy'], y_values_copy['energy']
            ys['by'], ys['by_err'] = y_values_copy['bjorkeny'], y_values_copy['bjorkeny']

            ys['vx'], ys['vx_err'] = y_values_copy['vertex_pos_x'], y_values_copy['vertex_pos_x']
            ys['vy'], ys['vy_err'] = y_values_copy['vertex_pos_y'], y_values_copy['vertex_pos_y']
            ys['vz'], ys['vz_err'] = y_values_copy['vertex_pos_z'], y_values_copy['vertex_pos_z']
            ys['vt'], ys['vt_err'] = y_values_copy['time_residual_vertex'], y_values_copy['time_residual_vertex']

            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
            return ys

    elif name == 'ts_classifier':
        def label_modifier(y_values):
            # for every sample, [0,1] for shower, or [1,0] for track

            # {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}
            # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
            # label is always shower, except if muon-CC
            ys = dict()
            particle_type, is_cc = y_values['particle_type'], y_values['is_cc']
            is_muon_cc = np.logical_and(np.abs(particle_type) == 14, is_cc == 1)
            is_not_muon_cc = np.invert(is_muon_cc)

            batchsize = y_values.shape[0]
            # categorical [shower, track] -> [1,0] = shower, [0,1] = track
            categorical_ts = np.zeros((batchsize, 2), dtype='bool')

            categorical_ts[:, 0] = is_not_muon_cc
            categorical_ts[:, 1] = is_muon_cc

            ys['ts_output'] = categorical_ts.astype(np.float32)
            return ys

    elif name == 'bg_classifier':
        def label_modifier(y_values):
            # for every sample, [1,0,0] for neutrinos, [0,1,0] for mupage
            # and [0,0,1] for random_noise
            # particle types: mupage: np.abs(13), random_noise = 0, neutrinos =
            ys = dict()
            particle_type = y_values['particle_type']
            is_mupage = np.abs(particle_type) == 13
            is_random_noise = np.abs(particle_type == 0)
            is_not_mupage_nor_rn = np.invert(np.logical_or(is_mupage,
                                                           is_random_noise))

            batchsize = y_values.shape[0]
            categorical_bg = np.zeros((batchsize, 3), dtype='bool')

            categorical_bg[:, 0] = is_not_mupage_nor_rn
            categorical_bg[:, 1] = is_mupage
            categorical_bg[:, 2] = is_random_noise

            ys['bg_output'] = categorical_bg.astype(np.float32)
            return ys

    elif name == 'bg_classifier_2_class':
        def label_modifier(y_values):
            # for every sample, [1,0,0] for neutrinos, [0,1,0] for mupage
            # and [0,0,1] for random_noise
            # particle types: mupage: np.abs(13), random_noise = 0, neutrinos =
            ys = dict()
            particle_type = y_values['particle_type']
            is_mupage = np.abs(particle_type) == 13
            is_random_noise = np.abs(particle_type == 0)
            is_not_mupage_nor_rn = np.invert(np.logical_or(is_mupage,
                                                           is_random_noise))

            batchsize = y_values.shape[0]
            categorical_bg = np.zeros((batchsize, 2), dtype='bool')

            # neutrino
            categorical_bg[:, 0] = is_not_mupage_nor_rn
            # is not neutrino
            categorical_bg[:, 1] = np.invert(is_not_mupage_nor_rn)

            ys['bg_output'] = categorical_bg.astype(np.float32)
            return ys

    else:
        raise ValueError("Unknown output_type: " + str(name))

    return label_modifier


def orca_dataset_modifiers(name):
    """
    Returns one of the dataset modifiers used for predicting with OrcaNet.

    Parameters
    ----------
    name : str
        Name of the dataset modifier that should be used.

    """
    if name == "struc_arr":
        # Multi-purpose conversion to rec array
        #
        # Output from network: Dict with 2darrays, shapes (x, y_i)
        # Transform this into a recarray with shape (x, y_1 + y_2 + ...) like this:
        # y_pred = {"foo": ndarray, "bar": ndarray}
        # --> dtypes = [foo_1, foo_2, ..., bar_1, bar_2, ... ]

        def dataset_modifier(info_blob):
            y_pred = info_blob["y_pred"]
            y_true = info_blob["y_true"]
            y_values = info_blob["y_values"]
            datasets = dict()
            datasets["pred"] = dict_to_recarray(y_pred)

            if y_true is not None:
                datasets["true"] = dict_to_recarray(y_true)

            if y_values is not None:
                datasets['mc_info'] = y_values  # is already a structured array

            return datasets

    elif name == 'bg_classifier':
        def dataset_modifier(mc_info, y_true, y_pred):

            # y_pred and y_true are dicts with keys for each output
            # we only have 1 output in case of the bg classifier
            y_pred = y_pred['bg_output']
            y_true = y_true['bg_output']

            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            # make pred dataset
            dtypes = np.dtype([('prob_neutrino', y_pred.dtype),
                               ('prob_muon', y_pred.dtype),
                               ('prob_random_noise', y_pred.dtype)])
            pred = np.empty(y_pred.shape[0], dtype=dtypes)
            pred['prob_neutrino'] = y_pred[:, 0]
            pred['prob_muon'] = y_pred[:, 1]
            pred['prob_random_noise'] = y_pred[:, 2]

            datasets['pred'] = pred

            # make true dataset
            dtypes = np.dtype([('cat_neutrino', y_true.dtype),
                               ('cat_muon', y_true.dtype),
                               ('cat_random_noise', y_true.dtype)])
            true = np.empty(y_true.shape[0], dtype=dtypes)
            true['cat_neutrino'] = y_true[:, 0]
            true['cat_muon'] = y_true[:, 1]
            true['cat_random_noise'] = y_true[:, 2]

            datasets['true'] = true

            return datasets

    elif name == 'bg_classifier_2_class':
        def dataset_modifier(mc_info, y_true, y_pred):

            # y_pred and y_true are dicts with keys for each output
            # we only have 1 output in case of the bg classifier
            y_pred = y_pred['bg_output']
            y_true = y_true['bg_output']

            datasets = dict()  # y_pred is a list of arrays
            datasets['mc_info'] = mc_info  # is already a structured array

            # make pred dataset
            dtypes = np.dtype([('prob_neutrino', y_pred.dtype),
                               ('prob_not_neutrino', y_pred.dtype)])
            pred = np.empty(y_pred.shape[0], dtype=dtypes)
            pred['prob_neutrino'] = y_pred[:, 0]
            pred['prob_not_neutrino'] = y_pred[:, 1]

            datasets['pred'] = pred

            # make true dataset
            dtypes = np.dtype([('cat_neutrino', y_true.dtype),
                               ('cat_not_neutrino', y_true.dtype)])
            true = np.empty(y_true.shape[0], dtype=dtypes)
            true['cat_neutrino'] = y_true[:, 0]
            true['cat_not_neutrino'] = y_true[:, 1]

            datasets['true'] = true

            return datasets

    elif name == 'ts_classifier':
        def dataset_modifier(mc_info, y_true, y_pred):

            # y_pred and y_true are dicts with keys for each output
            # we only have 1 output in case of the ts classifier
            y_pred = y_pred['ts_output']
            y_true = y_true['ts_output']

            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            # make pred dataset
            dtypes = np.dtype([('prob_shower', y_pred.dtype),
                               ('prob_track', y_pred.dtype)])
            pred = np.empty(y_pred.shape[0], dtype=dtypes)
            pred['prob_shower'] = y_pred[:, 0]
            pred['prob_track'] = y_pred[:, 1]

            datasets['pred'] = pred

            # make true dataset
            dtypes = np.dtype([('cat_shower', y_true.dtype),
                               ('cat_track', y_true.dtype)])
            true = np.empty(y_true.shape[0], dtype=dtypes)
            true['cat_shower'] = y_true[:, 0]
            true['cat_track'] = y_true[:, 1]

            datasets['true'] = true

            return datasets

    elif name == 'regression_energy_dir_bjorken-y_vtx_errors':
        def dataset_modifier(mc_info, y_true, y_pred):

            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""

            pred_labels_and_nn_output_names = [('pred_energy', 'e'), ('pred_dir_x', 'dx'), ('pred_dir_y', 'dy'),
                                               ('pred_dir_z', 'dz'), ('pred_bjorkeny', 'by'), ('pred_vtx_x', 'vx'),
                                               ('pred_vtx_y', 'vy'), ('pred_vtx_z', 'vz'), ('pred_vtx_t', 'vt'),
                                               ('pred_err_energy', 'e_err'), ('pred_err_dir_x', 'dx_err'),
                                               ('pred_err_dir_y', 'dy_err'), ('pred_err_dir_z', 'dz_err'),
                                               ('pred_err_bjorkeny', 'by_err'), ('pred_err_vtx_x', 'vx_err'),
                                               ('pred_err_vtx_y', 'vy_err'), ('pred_err_vtx_z', 'vz_err'),
                                               ('pred_err_vtx_t', 'vt_err')]

            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['e'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                if 'err' in tpl[1]:
                    # the err outputs have shape (bs, 2) with 2 (pred_label, pred_label_err)
                    # we only want to select the pred_label_err output
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                else:
                    pred[tpl[0]] = np.squeeze(y_pred[tpl[1]], axis=1)  # reshape (bs, 1) to (bs)

            datasets['pred'] = pred

            # make true dataset
            true_labels_and_nn_output_names = [('true_energy', 'e'), ('true_dir_x', 'dx'), ('true_dir_y', 'dy'),
                                               ('true_dir_z', 'dz'), ('true_bjorkeny', 'by'), ('true_vtx_x', 'vx'),
                                               ('true_vtx_y', 'vy'), ('true_vtx_z', 'vz'), ('true_vtx_t', 'vt'),
                                               ('true_err_energy', 'e_err'), ('true_err_dir_x', 'dx_err'),
                                               ('true_err_dir_y', 'dy_err'), ('true_err_dir_z', 'dz_err'),
                                               ('true_err_bjorkeny', 'by_err'), ('true_err_vtx_x', 'vx_err'),
                                               ('true_err_vtx_y', 'vy_err'), ('true_err_vtx_z', 'vz_err'),
                                               ('true_err_vtx_t', 'vt_err')]

            dtypes_true = [(tpl[0], y_true[tpl[1]].dtype) for tpl in true_labels_and_nn_output_names]
            true = np.empty(n_evts, dtype=dtypes_true)

            for tpl in true_labels_and_nn_output_names:
                true[tpl[0]] = y_true[tpl[1]]

            datasets['true'] = true

            return datasets

    else:
        raise ValueError('Unknown dataset modifier: ' + str(name))

    return dataset_modifier


def dict_to_recarray(data_dict):
    """
    Convert a dict with 2d np arrays to a 2d struc array, with column
    names derived from the dict keys.

    Parameters
    ----------
    data_dict : dict
        Keys: name of the output layer.
        Values: 2d arrays, first dimension matches

    Returns
    -------
    recarray : ndarray

    """
    column_names = []
    for output_name, data in data_dict.items():
        columns = data.shape[1]
        for i in range(columns):
            column_names.append(output_name + "_" + str(i+1))
    names = ",".join([name for name in column_names])

    data = np.concatenate(list(data_dict.values()), axis=1)
    recarray = np.core.records.fromrecords(data, names=names)
    return recarray


def orca_learning_rates(name, total_file_no):
    """
    Returns one of the learning rate schedules used for Orca networks.

    Parameters
    ----------
    name : str
        Name of the schedule.
    total_file_no : int
        How many files there are to train on.

    Returns
    -------
    learning_rate : function
        The learning rate schedule.

    """
    if name == "triple_decay":
        def learning_rate(n_epoch, n_file):
            """
            Function that calculates the current learning rate based on
            the number of already trained epochs.

            Learning rate schedule: lr_decay = 7% for lr > 0.0003
                                    lr_decay = 4% for 0.0003 >= lr > 0.0001
                                    lr_decay = 2% for 0.0001 >= lr

            Parameters
            ----------
            n_epoch : int
                The number of the current epoch which is used to calculate
                the new learning rate.
            n_file : int
                The number of the current filenumber which is used to
                calculate the new learning rate.

            Returns
            -------
            lr_temp : float
                Calculated learning rate for this epoch.

            """
            n_lr_decays = (n_epoch - 1) * total_file_no + (n_file - 1)
            lr_temp = 0.005  # * n_gpu TODO think about multi gpu lr

            for i in range(n_lr_decays):
                if lr_temp > 0.0003:
                    lr_decay = 0.07  # standard for regression: 0.07, standard for PID: 0.02
                elif 0.0003 >= lr_temp > 0.0001:
                    lr_decay = 0.04  # standard for regression: 0.04, standard for PID: 0.01
                else:
                    lr_decay = 0.02  # standard for regression: 0.02, standard for PID: 0.005
                lr_temp = lr_temp * (1 - float(lr_decay))

            return lr_temp

    elif name == "triple_decay_weaker":
        def learning_rate(n_epoch, n_file):
            """
            Function that calculates the current learning rate based on
            the number of already trained epochs.

            Learning rate schedule: lr_decay = 2% for lr > 0.0003
                                    lr_decay = 1% for 0.0003 >= lr > 0.0001
                                    lr_decay = 0.5% for 0.0001 >= lr

            Parameters
            ----------
            n_epoch : int
                The number of the current epoch which is used to calculate
                the new learning rate.
            n_file : int
                The number of the current filenumber which is used to
                calculate the new learning rate.

            Returns
            -------
            lr_temp : float
                Calculated learning rate for this epoch.

            """
            n_lr_decays = (n_epoch - 1) * total_file_no + (n_file - 1)
            lr_temp = 0.003  # * n_gpu TODO think about multi gpu lr

            for i in range(n_lr_decays):
                if lr_temp > 0.0003:
                    lr_decay = 0.02  # standard for regression: 0.07, standard for PID: 0.02
                elif 0.0003 >= lr_temp > 0.0001:
                    lr_decay = 0.01  # standard for regression: 0.04, standard for PID: 0.01
                else:
                    lr_decay = 0.005  # standard for regression: 0.02, standard for PID: 0.005
                lr_temp = lr_temp * (1 - float(lr_decay))

            return lr_temp

    else:
        raise NameError("Unknown orca learning rate name", name)

    return learning_rate
