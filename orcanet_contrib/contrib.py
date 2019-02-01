import numpy as np


def orca_sample_modifiers(swap_col, str_ident):
    """
    Returns one of the sample modifiers used for Orca networks.

    They will permute columns, and/or add permuted columns to xs.

    The input to the functions is:
        xs_list : dict
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
    swap_col : None/str
        Define which channels to swap.
    str_ident : str
        Additional operations.

    Returns
    -------
    sample_modifier : function
        The sample modifier function.

    """
    swap_4d_channels_dict = {'yzt-x': (0, 2, 3, 4, 1), 'xyt-z': (0, 1, 2, 4, 3), 't-xyz': (0, 4, 1, 2, 3),
                             'tyz-x': (0, 4, 2, 3, 1)}

    if swap_col in swap_4d_channels_dict:
        def swap_columns(xs_list):
            # Transpose dimensions
            xs_layer = dict()
            keys = list(xs_list.keys())
            xs_layer[keys[0]] = np.transpose(xs_list, swap_4d_channels_dict[swap_col])
            return xs_layer
        sample_modifier = swap_columns

    elif swap_col == 'xyz-t_and_yzt-x':
        def sample_modifier(xs_list):
            # Use xyz-t, and also transpose it to yzt-x and use that, too.
            xs_layer = dict()
            xs_layer["xyz-t"] = xs_list["xyz-t"]
            xs_layer["yzt-x"] = np.transpose(xs_list["xyz-t"], swap_4d_channels_dict['yzt-x'])
            return xs_layer

    elif 'xyz-t_and_yzt-x' + 'multi_input_single_train_tight-1_tight-2' in swap_col + str_ident:
        def sample_modifier(xs_list):
            # Use xyz-t in two different time cuts, and also transpose them to yzt-x and use these, too.
            xs_layer = dict()
            xs_layer["xyz-t_tight-1"] = xs_list["xyz-t_tight-1"]
            xs_layer["xyz-t_tight-2"] = xs_list["xyz-t_tight-2"]
            xs_layer["yzt-x_tight-1"] = np.transpose(xs_list["xyz-t_tight-1"], swap_4d_channels_dict['yzt-x'])
            xs_layer["yzt-x_tight-2"] = np.transpose(xs_list["xyz-t_tight-2"], swap_4d_channels_dict['yzt-x'])
            return xs_layer

    elif swap_col == 'xyz-t_and_xyz-c_single_input':
        def sample_modifier(xs_list):
            # Concatenate xyz-t and xyz-c to a single input
            xs_layer = dict()
            xs_layer['xyz-t_and_xyz-c_single_input'] = np.concatenate([xs_list["xyz-t"], xs_list["xyz-c"]], axis=-1)
            return xs_layer

    else:
        raise ValueError('The argument "swap_col"=' + str(swap_col) + ' is not valid.')

    return sample_modifier


def orca_label_modifiers(class_type):
    """
    TODO add docs

    CAREFUL: y_values is a structured numpy array! if you use advanced numpy indexing, this may lead to errors.
    Let's suppose you want to assign a particular value to one or multiple elements of the y_values array.

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
    class_type : str

    Returns
    -------
    label_modifier : function
        The label modifier function.

    """

    if class_type == 'energy_dir_bjorken-y_vtx_errors':
        def label_modifier(y_values):
            ys = dict()
            particle_type, is_cc = y_values['particle_type'], y_values['is_cc']
            elec_nc_bool_idx = np.logical_and(np.abs(particle_type) == 12, is_cc == 0)

            # correct energy to visible energy
            visible_energy = y_values[elec_nc_bool_idx]['energy'] * y_values[elec_nc_bool_idx]['bjorkeny']
            # fix energy to visible energy
            np.place(y_values['energy'], elec_nc_bool_idx, visible_energy)  # TODO fix numpy warning
            # set bjorkeny label of nc events to 1
            np.place(y_values['bjorkeny'], elec_nc_bool_idx, 1)

            ys['dx'], ys['dx_err'] = y_values['dir_x'], y_values['dir_x']
            ys['dy'], ys['dy_err'] = y_values['dir_y'], y_values['dir_y']
            ys['dz'], ys['dz_err'] = y_values['dir_z'], y_values['dir_z']
            ys['e'], ys['e_err'] = y_values['energy'], y_values['energy']
            ys['by'], ys['by_err'] = y_values['bjorkeny'], y_values['bjorkeny']

            ys['vx'], ys['vx_err'] = y_values['vertex_pos_x'], y_values['vertex_pos_x']
            ys['vy'], ys['vy_err'] = y_values['vertex_pos_y'], y_values['vertex_pos_y']
            ys['vz'], ys['vz_err'] = y_values['vertex_pos_z'], y_values['vertex_pos_z']
            ys['vt'], ys['vt_err'] = y_values['time_residual_vertex'], y_values['time_residual_vertex']

            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
            return ys

    elif class_type == 'track-shower':
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
            categorical_ts = np.zeros((batchsize, 2), dtype='bool')  # categorical [shower, track] -> [1,0] = shower, [0,1] = track
            categorical_ts[is_not_muon_cc][:, 0] = 1
            categorical_ts[is_muon_cc][:, 1] = 1

            ys['ts_output'] = categorical_ts.astype(np.float32)
            return ys

    else:
        raise ValueError('The label ' + str(class_type) + ' in class_type is not available.')

    return label_modifier


def orca_learning_rates(name):
    """
    Learning rate schedules used for ORCA.

    Parameters
    ----------
    name : str
        Name of the schedule.

    Returns
    -------
    learning_rate : function
        The learning rate schedule.

    """
    if name == "triple_decay":
        def learning_rate(n_epoch, n_file, cfg):
            """
            Function that calculates the current learning rate based on the number of already trained epochs.

            Learning rate schedule is as follows: lr_decay = 7% for lr > 0.0003
                                                  lr_decay = 4% for 0.0003 >= lr > 0.0001
                                                  lr_decay = 2% for 0.0001 >= lr

            Parameters
            ----------
            n_epoch : int
                The number of the current epoch which is used to calculate the new learning rate.
            n_file : int
                The number of the current filenumber which is used to calculate the new learning rate.
            cfg : object Configuration
                Configuration object containing all the configurable options in the OrcaNet scripts.

            Returns
            -------
            lr_temp : float
                Calculated learning rate for this epoch.

            """
            n_lr_decays = (n_epoch - 1) * cfg.get_no_of_files(for_val_files=False) + (n_file - 1)
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
    else:
        raise NameError("Unknown orca learning rate name", name)

    return learning_rate
