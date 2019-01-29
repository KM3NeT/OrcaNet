import numpy as np


def sample_modifiers_orca(swap_col, str_ident):
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
