#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Michael's orcanet utility stuff.

"""
import warnings
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

class GraphSampleMod:
    """
    Read out points, coordinates and is_valid from the ndarray h5 set.

    Attributes
    ----------
    preproc_knn : int, optional
        Do the knn operations. Returns dict with 'xixj' in this case.
    with_lightspeed : bool
        Multiply time with lightspeed.
    with_n_hits : int
        If 1, also get the number of hits. If 2, get n_hits but dont whiten.
    knn : int
        Skip batches with events that have to few hits for given knn.

    """
    def __init__(self, preproc_knn=None, with_lightspeed=True, with_n_hits=0,
                 knn=16):
        self.preproc_knn = preproc_knn
        self.with_lightspeed = with_lightspeed
        self.with_n_hits = with_n_hits
        self.knn = knn
            
        #old one
        self.column_names = (
           'channel_id', 'dir_x', 'dir_y', 'dir_z',
           'dom_id', 'du', 'floor', 'group_id',
           'pos_x', 'pos_y', 'pos_z', 't0', 'time',
           'tot', 'triggered', 'is_valid')
           #self.column_names = ("pos_x", "pos_y", "pos_z",
           # "time", "dir_x", "dir_y", "dir_z", "is_valid")
                 
        self.lightspeed = 0.225  # in water; m/ns

    @classmethod
    def from_str(cls, string):
        """ E.g. 'preproc_knn=5,with_lightspeed=1' """
        kwargs = {}
        for arg in string.split(","):
            name, value = arg.split("=")
            kwargs[name] = int(value)
        return cls(**kwargs)

    def _str_to_idx(self, which):
        if isinstance(which, str):
            return self.column_names.index(which)
        else:
            return [self.column_names.index(w) for w in which]

    def __call__(self, info_blob):

        points = info_blob["x_values"]["points"]

        for_nodes = ("pos_x", "pos_y", "pos_z", "time", "dir_x", "dir_y", "dir_z")
        for_coords = ("pos_x", "pos_y", "pos_z", "time")
        for_valid = "is_valid"
        
        nodes = points[:, :, self._str_to_idx(for_nodes)].astype("float32")
        coords = points[:, :, self._str_to_idx(for_coords)].astype("float32")
        if self.with_lightspeed:
            coords[:, :, -1] *= self.lightspeed
        is_valid = points[:, :, self._str_to_idx(for_valid)].astype("float32")
        
        # pad events with less then 17 hits (for 16 knn) by duping first hit
        if self.knn is not None:
            min_n_hits = self.knn + 1
            n_hits = is_valid.sum(axis=-1)
            too_small = n_hits < min_n_hits
            if any(too_small):
                #warnings.warn(f"Too few hits! Needed {min_n_hits}, "
                #              f"had {n_hits[too_small]}! Padding...")
                for event_no in np.where(too_small)[0]:
                    hits = int(n_hits[event_no])
                    is_valid[event_no, hits:min_n_hits] = 1.
                    nodes[event_no, hits:min_n_hits] = nodes[event_no, 0]
                    coords[event_no, hits:min_n_hits] = coords[event_no, 0]

        xs = {
            "nodes": nodes,
            "is_valid": is_valid,
            "coords": coords,
        }
        if self.preproc_knn:
            xi, xj = _get_xixj(**xs, k=self.preproc_knn)
            xs = {
                "nodes": nodes,
                "is_valid": is_valid,
                "xi": xi,
                "xj": xj,
            }
        if self.with_n_hits > 0:
            n_hits = info_blob["y_values"]["n_hits"]
            # take log and whiten
            if self.with_n_hits == 1:
                n_hits = (np.log(n_hits) - 4.557106)/0.46393168
            xs["n_hits"] = np.expand_dims(n_hits, -1).astype("float32")
        return xs

 

class GraphSampleMod_with_trig:
    """
    Read out points, coordinates and is_valid from the ndarray h5 set.

    Attributes
    ----------
    preproc_knn : int, optional
        Do the knn operations. Returns dict with 'xixj' in this case.
    with_lightspeed : bool
        Multiply time with lightspeed.
    with_n_hits : int
        If 1, also get the number of hits. If 2, get n_hits but dont whiten.
    knn : int
        Skip batches with events that have to few hits for given knn.

    """
    def __init__(self, preproc_knn=None, with_lightspeed=True, with_n_hits=0,
                 knn=16):
        self.preproc_knn = preproc_knn
        self.with_lightspeed = with_lightspeed
        self.with_n_hits = with_n_hits
        self.knn = knn
            
        #old one
        self.column_names = (
           'channel_id', 'dir_x', 'dir_y', 'dir_z',
           'dom_id', 'du', 'floor', 'group_id',
           'pos_x', 'pos_y', 'pos_z', 't0', 'time',
           'tot', 'triggered', 'is_valid')
           #self.column_names = ("pos_x", "pos_y", "pos_z",
           # "time", "dir_x", "dir_y", "dir_z", "is_valid")
                 
        self.lightspeed = 0.225  # in water; m/ns

    @classmethod
    def from_str(cls, string):
        """ E.g. 'preproc_knn=5,with_lightspeed=1' """
        kwargs = {}
        for arg in string.split(","):
            name, value = arg.split("=")
            kwargs[name] = int(value)
        return cls(**kwargs)

    def _str_to_idx(self, which):
        if isinstance(which, str):
            return self.column_names.index(which)
        else:
            return [self.column_names.index(w) for w in which]

    def __call__(self, info_blob):

        points = info_blob["x_values"]["points"]

        for_nodes = ("pos_x", "pos_y", "pos_z", "time", "dir_x", "dir_y", "dir_z", "triggered")
        for_coords = ("pos_x", "pos_y", "pos_z", "time")
        for_valid = "is_valid"
        
        nodes = points[:, :, self._str_to_idx(for_nodes)].astype("float32")
        coords = points[:, :, self._str_to_idx(for_coords)].astype("float32")
        if self.with_lightspeed:
            coords[:, :, -1] *= self.lightspeed
        is_valid = points[:, :, self._str_to_idx(for_valid)].astype("float32")
        
        # pad events with less then 17 hits (for 16 knn) by duping first hit
        if self.knn is not None:
            min_n_hits = self.knn + 1
            n_hits = is_valid.sum(axis=-1)
            too_small = n_hits < min_n_hits
            if any(too_small):
                #warnings.warn(f"Too few hits! Needed {min_n_hits}, "
                #              f"had {n_hits[too_small]}! Padding...")
                for event_no in np.where(too_small)[0]:
                    hits = int(n_hits[event_no])
                    is_valid[event_no, hits:min_n_hits] = 1.
                    nodes[event_no, hits:min_n_hits] = nodes[event_no, 0]
                    coords[event_no, hits:min_n_hits] = coords[event_no, 0]

        xs = {
            "nodes": nodes,
            "is_valid": is_valid,
            "coords": coords,
        }
        if self.preproc_knn:
            xi, xj = _get_xixj(**xs, k=self.preproc_knn)
            xs = {
                "nodes": nodes,
                "is_valid": is_valid,
                "xi": xi,
                "xj": xj,
            }
        if self.with_n_hits > 0:
            n_hits = info_blob["y_values"]["n_hits"]
            # take log and whiten
            if self.with_n_hits == 1:
                n_hits = (np.log(n_hits) - 4.557106)/0.46393168
            xs["n_hits"] = np.expand_dims(n_hits, -1).astype("float32")
        return xs


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
                [xs_files["x_values"]['xyz-t'], xs_files["x_values"]['xyz-c']], axis=-1)
            return xs_layer  

    elif name == 'xyz-t_and_xyz-c_boosted_c':
        def sample_modifier(xs_files):
            # Concatenate xyz-t and xyz-c to a single input
            xs_layer = dict()
            xs_layer['xyz-t_and_xyz-c_single_input'] = np.concatenate(
                [xs_files["x_values"]['xyz-t'], xs_files["x_values"]['xyz-c'], xs_files["x_values"]['xyz-c'],
                 xs_files["x_values"]['xyz-c'], xs_files["x_values"]['xyz-c'], xs_files["x_values"]['xyz-c'],
                  xs_files["x_values"]['xyz-c'], xs_files["x_values"]['xyz-c'], xs_files["x_values"]['xyz-c'],
                   xs_files["x_values"]['xyz-c'], xs_files["x_values"]['xyz-c']
                ], axis=-1)
        
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
    
    if name == 'dz_error':
        def label_modifier(data):
            
            y_values = data["y_values"]

            ys = dict()
            
            # make a copy of the y_values array, since we modify it now
            y_values_copy = np.copy(y_values)
            
            #ys['dz'],ys['dz_err'] = y_values_copy['dir_z'],y_values_copy['dir_z']
            ys['dz'] = y_values_copy['dir_z']
            
            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
                
            return ys 

    elif name == 'dir':
        def label_modifier(data):
            
            y_values = data["y_values"]

            ys = dict()
            
            # make a copy of the y_values array, since we modify it now
            y_values_copy = np.copy(y_values)
            
            #ys['dz'],ys['dz_err'] = y_values_copy['dir_z'],y_values_copy['dir_z']
            ys['dx'] = y_values_copy['dir_x']
            ys['dy'] = y_values_copy['dir_y']
            ys['dz'] = y_values_copy['dir_z']
            
            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
                
            return ys 

    elif name == 'pos':
        def label_modifier(data):
            
            y_values = data["y_values"]

            ys = dict()
            
            # make a copy of the y_values array, since we modify it now
            y_values_copy = np.copy(y_values)
            
            #ys['dz'],ys['dz_err'] = y_values_copy['dir_z'],y_values_copy['dir_z']
            ys['vx'] = y_values_copy['vertex_pos_x']
            ys['vy'] = y_values_copy['vertex_pos_y']
            ys['vz'] = y_values_copy['vertex_pos_z']
            
            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
                
            return ys 
                            
    elif name == 'e_error':
        def label_modifier(data):
            
            y_values = data["y_values"]

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
            
            #ys['e'], ys['e_err'] = np.log(y_values_copy['energy']), np.log(y_values_copy['energy'])
            ys['e'] = np.log(y_values_copy['energy'])
                        
            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
                
            return ys 
                 
    elif name == 'dz_e':
        def label_modifier(data):
            
            y_values = data["y_values"]

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
            
            ys['dz'] = y_values_copy['dir_z']            
            ys['e'] = np.log(y_values_copy['energy'])
            
            for key_label in ys:
                ys[key_label] = ys[key_label].astype(np.float32)
                
            return ys 
            
    elif name == 'energy_dir_bjorken-y_vtx_errors':
        def label_modifier(data):
            
            y_values = data["y_values"]
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
        def label_modifier(data):
            
            y_values = data["y_values"]
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

    elif name == 'bg_classifier_3_class':
        def label_modifier(data):
            
            y_values = data["y_values"]
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

    elif name == 'bg_classifier_4_class':
        def label_modifier(data):
            
            y_values = data["y_values"]
            # for every sample, [1,0,0] for neutrinos, [0,1,0] for mupage
            # and [0,0,1] for random_noise - only with 4 classes
            # particle types: mupage: np.abs(13), random_noise = 0, neutrinos =
            ys = dict()
            particle_type = y_values['particle_type']
            is_mupage = np.abs(particle_type) == 13
            is_random_noise = np.abs(particle_type == 0)
            
            is_muon_neutrino = np.abs(particle_type) == 14
            is_electron_neutrino = np.abs(particle_type) == 12
            is_cc = y_values['is_cc']
            
            is_neutrino_track = np.logical_and(is_muon_neutrino,is_cc)
            is_nc_muon_neutrino = np.logical_and(is_muon_neutrino,np.invert(is_cc))
            is_neutrino_shower = np.logical_or(is_electron_neutrino,is_nc_muon_neutrino)
                                    
            batchsize = y_values.shape[0]
            categorical_bg = np.zeros((batchsize, 4), dtype='bool')

            categorical_bg[:, 0] = is_neutrino_track            
            categorical_bg[:, 1] = is_neutrino_shower
            categorical_bg[:, 2] = is_mupage
            categorical_bg[:, 3] = is_random_noise

            ys['bg_output'] = categorical_bg.astype(np.float32)
            return ys
            
    elif name == 'bg_classifier_2_class':
        def label_modifier(data):
            
            y_values = data["y_values"]
            
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

    elif name == 'real_data':
        def label_modifier(data):
            
            #do nothing here
            ys = dict()

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

            
    elif name == 'bg_classifier_3_class':
        def dataset_modifier(info_blob):

            #blob contains: y_values (mc info), xs ("images/graphs"), ys (true labels), y_pred (predicted labels)
        
            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            y_pred = y_pred['bg_output']
            y_true = y_true['bg_output']

            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array
            
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
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

    elif name == 'bg_classifier_4_class':
        def dataset_modifier(info_blob):

            #blob contains: y_values (mc info), xs ("images/graphs"), ys (true labels), y_pred (predicted labels)
        
            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            y_pred = y_pred['bg_output']
            y_true = y_true['bg_output']

            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array
            
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            dtypes = np.dtype([('prob_neutrino_track', y_pred.dtype),
                               ('prob_neutrino_shower', y_pred.dtype),
                               ('prob_muon', y_pred.dtype),
                               ('prob_random_noise', y_pred.dtype)])
            pred = np.empty(y_pred.shape[0], dtype=dtypes)
            pred['prob_neutrino_track'] = y_pred[:, 0]
            pred['prob_neutrino_shower'] = y_pred[:, 1]
            pred['prob_muon'] = y_pred[:, 2]
            pred['prob_random_noise'] = y_pred[:, 3]

            datasets['pred'] = pred

            # make true dataset
            dtypes = np.dtype([('cat_neutrino_track', y_true.dtype),
                               ('cat_neutrino_shower', y_true.dtype),
                               ('cat_muon', y_true.dtype),
                               ('cat_random_noise', y_true.dtype)])
            true = np.empty(y_true.shape[0], dtype=dtypes)
            true['cat_neutrino_track'] = y_true[:, 0]
            true['cat_neutrino_shower'] = y_true[:, 1]
            true['cat_muon'] = y_true[:, 2]
            true['cat_random_noise'] = y_true[:, 3]

            datasets['true'] = true

            return datasets

    elif name == 'bg_classifier_2_class':
        def dataset_modifier(info_blob):
            
            #blob contains: y_values (mc info), xs ("images/graphs"), ys (true labels), y_pred (predicted labels)
        
            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            y_pred = y_pred['bg_output']
            y_true = y_true['bg_output']
                        
            datasets = dict()  # y_pred is a list of arrays
            datasets['mc_info'] = mc_info  # is already a structured array
            
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
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

    elif name == 'bg_classifier_2_class_real_data':
        def dataset_modifier(info_blob):
            
            #blob contains: y_values (mc info), xs ("images"), ys (true labels), y_pred (predicted labels)
            
            #get info that was defined in the mc__info_extr in orcasong
            info = info_blob['y_values']
            
            #get info from the prediction
            y_pred = info_blob['y_pred']
            y_pred = y_pred['bg_output']
            
            datasets = dict()  # y_pred is a list of arrays
            datasets['info'] = info  # is already a structured array
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
                        
            # make pred dataset
            dtypes = np.dtype([('prob_neutrino', y_pred.dtype),
                               ('prob_not_neutrino', y_pred.dtype)])
            pred = np.empty(y_pred.shape[0], dtype=dtypes)
            pred['prob_neutrino'] = y_pred[:, 0]
            pred['prob_not_neutrino'] = y_pred[:, 1]

            datasets['pred'] = pred

            return datasets
    
            

    elif name == 'ts_classifier':
        def dataset_modifier(info_blob):

            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            # y_pred and y_true are dicts with keys for each output
            # we only have 1 output in case of the ts classifier
            y_pred = y_pred['ts_output']
            y_true = y_true['ts_output']
            
            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]

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

    elif name == 'ts_classifier_real_data':
        def dataset_modifier(info_blob):

            info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            
            y_pred = y_pred['ts_output']
            
            datasets = dict()
            datasets['info'] = info  # is already a structured array
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            dtypes = np.dtype([('prob_shower', y_pred.dtype),
                               ('prob_track', y_pred.dtype)])
            pred = np.empty(y_pred.shape[0], dtype=dtypes)
            pred['prob_shower'] = y_pred[:, 0]
            pred['prob_track'] = y_pred[:, 1]

            datasets['pred'] = pred

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

    elif name == 'regression_dz_error':
        def dataset_modifier(info_blob):

            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""

            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_dir_z', 'dz'),
                                               ('pred_dir_z_err','dz')
                                                ]
            
            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['dz'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                #the uncertainty estimation is the 2nd column
                if 'err' in tpl[0]:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                #the fitted value is the first
                else:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 0]

            datasets['pred'] = pred

            # make true dataset
            true_labels_and_nn_output_names = [('true_dir_z', 'dz'),
                                               ]

            dtypes_true = [(tpl[0], y_true[tpl[1]].dtype) for tpl in true_labels_and_nn_output_names]
            true = np.empty(n_evts, dtype=dtypes_true)

            for tpl in true_labels_and_nn_output_names:
                true[tpl[0]] = y_true[tpl[1]]

            datasets['true'] = true
            
            return datasets

    elif name == 'regression_dir':
        def dataset_modifier(info_blob):

            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""

            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_dir_x', 'dx'),
                                               ('pred_dir_x_err','dx'),
                                               ('pred_dir_y', 'dy'),
                                               ('pred_dir_y_err','dy'),
                                               ('pred_dir_z', 'dz'),
                                               ('pred_dir_z_err','dz')
                                                ]
            
            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['dz'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                #the uncertainty estimation is the 2nd column
                if 'err' in tpl[0]:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                #the fitted value is the first
                else:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 0]

            datasets['pred'] = pred

            # make true dataset
            true_labels_and_nn_output_names = [('true_dir_x', 'dx'),
                                               ('true_dir_y', 'dy'),
                                               ('true_dir_z', 'dz'),
                                               ]

            dtypes_true = [(tpl[0], y_true[tpl[1]].dtype) for tpl in true_labels_and_nn_output_names]
            true = np.empty(n_evts, dtype=dtypes_true)

            for tpl in true_labels_and_nn_output_names:
                true[tpl[0]] = y_true[tpl[1]]

            datasets['true'] = true
            
            return datasets


    elif name == 'regression_pos':
        def dataset_modifier(info_blob):

            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
            
            datasets = dict()
            datasets['mc_info'] = mc_info  # is already a structured array

            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""

            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_pos_x', 'vx'),
                                               ('pred_pos_x_err','vx'),
                                               ('pred_pos_y', 'vy'),
                                               ('pred_pos_y_err','vy'),
                                               ('pred_pos_z', 'vz'),
                                               ('pred_pos_z_err','vz')
                                                ]
            
            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['vz'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                #the uncertainty estimation is the 2nd column
                if 'err' in tpl[0]:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                #the fitted value is the first
                else:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 0]

            datasets['pred'] = pred

            # make true dataset
            true_labels_and_nn_output_names = [('true_pos_x', 'vx'),
                                               ('true_pos_y', 'vy'),
                                               ('true_pos_z', 'vz'),
                                               ]

            dtypes_true = [(tpl[0], y_true[tpl[1]].dtype) for tpl in true_labels_and_nn_output_names]
            true = np.empty(n_evts, dtype=dtypes_true)

            for tpl in true_labels_and_nn_output_names:
                true[tpl[0]] = y_true[tpl[1]]

            datasets['true'] = true
            
            return datasets


    
    elif name == 'regression_e_error':
        def dataset_modifier(info_blob):

            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
        
            datasets = dict()
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            datasets['mc_info'] = mc_info  # is already a structured array
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""
            
            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_e', 'e'),
                                               ('pred_e_err','e')
                                                ]

            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['e'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                #the uncertainty estimation is the 2nd column
                if 'err' in tpl[0]:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                #the fitted value is the first
                else:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 0]

            datasets['pred'] = pred

            # make true dataset
            true_labels_and_nn_output_names = [('true_e', 'e'),
                                               #('true_e_err', 'e_err')
                                               ]

            dtypes_true = [(tpl[0], y_true[tpl[1]].dtype) for tpl in true_labels_and_nn_output_names]
            true = np.empty(n_evts, dtype=dtypes_true)

            for tpl in true_labels_and_nn_output_names:
                true[tpl[0]] = y_true[tpl[1]]

            datasets['true'] = true
            
            return datasets


    
    elif name == 'regression_dz_e':
        def dataset_modifier(info_blob):

            mc_info = info_blob['y_values']
            y_pred = info_blob['y_pred']
            y_true = info_blob['ys']
        
            datasets = dict()
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            datasets['mc_info'] = mc_info  # is already a structured array
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""

            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_dir_z', 'dz'),
                                               ('pred_dir_z_err', 'dz'),
                                               ('pred_e', 'e'),
                                               ('pred_e_err', 'e'),
                                                ]

            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['dz'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                #the uncertainty estimation is the 2nd column
                if 'err' in tpl[0]:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                #the fitted value is the first
                else:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 0]

            datasets['pred'] = pred

            # make true dataset
            true_labels_and_nn_output_names = [('true_dir_z', 'dz'),
                                                ('true_e', 'e'),
                                               ]

            dtypes_true = [(tpl[0], y_true[tpl[1]].dtype) for tpl in true_labels_and_nn_output_names]
            true = np.empty(n_evts, dtype=dtypes_true)

            for tpl in true_labels_and_nn_output_names:
                true[tpl[0]] = y_true[tpl[1]]

            datasets['true'] = true
            
            return datasets

            
    elif name == 'regression_dz_error_real_data':
        def dataset_modifier(info_blob):

            info = info_blob['y_values']
            y_pred = info_blob['y_pred']
                        
            datasets = dict()
            datasets['info'] = info  # is already a structured array
            
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""
            
            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_dir_z', 'dz'),
                                               ('pred_dir_z_err','dz')
                                                ]
                                                
            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['dz'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                if 'err' in tpl[1]:
                    # the err outputs have shape (bs, 2) with 2 (pred_label, pred_label_err)
                    # we only want to select the pred_label_err output
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1] 
                else:
                    pred[tpl[0]] = np.squeeze(y_pred[tpl[1]], axis=1)  # reshape (bs, 1) to (bs)

            datasets['pred'] = pred
 
            return datasets
   
    elif name == 'regression_dz_error_log_prob_real_data':
        def dataset_modifier(info_blob):

            info = info_blob['y_values']
            y_pred = info_blob['y_pred']
                        
            datasets = dict()
            datasets['info'] = info  # is already a structured array
            
            #add also the hit info
            datasets['hits'] = info_blob["x_values"]["points"]
            
            # make pred dataset
            """y_pred and y_true are dicts with keys for each output,
               here, we have 1 key for each regression variable"""
            
            #first entry of the tupel is the name for the output
            #second entry is how it was called in orcanet; take for both,
            #value and uncertainty the same variable name 
            pred_labels_and_nn_output_names = [('pred_dir_z', 'dz'),
                                               ('pred_dir_z_err','dz')
                                                ]
                                                
            dtypes_pred = [(tpl[0], y_pred[tpl[1]].dtype) for tpl in pred_labels_and_nn_output_names]
            n_evts = y_pred['dz'].shape[0]
            pred = np.empty(n_evts, dtype=dtypes_pred)

            for tpl in pred_labels_and_nn_output_names:
                #the uncertainty estimation is the 2nd column
                if 'err' in tpl[0]:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 1]
                #the fitted value is the first
                else:
                    pred[tpl[0]] = y_pred[tpl[1]][:, 0]

            datasets['pred'] = pred
 
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
            lr_temp = 0.05  # * n_gpu TODO think about multi gpu lr

            for i in range(n_lr_decays):
                if lr_temp > 0.03:
                    lr_decay = 0.07  # standard for regression: 0.07, standard for PID: 0.02
                elif 0.03 >= lr_temp > 0.01:
                    lr_decay = 0.04  # standard for regression: 0.04, standard for PID: 0.01
                else:
                    lr_decay = 0.02  # standard for regression: 0.02, standard for PID: 0.005
                lr_temp = lr_temp * (1 - float(lr_decay))

            return lr_temp
    
    elif name == "constant":
        def learning_rate(n_epoch, n_file):
            """
            Function that only returns a constant learning rate for test purposes

            """

            return 0.0003

    elif name == "triple_decay_weaker":
        def learning_rate(n_epoch, n_file):
            """
            Function that calculates the current learning rate based on
            the number of already trained epochs.

            Learning rate schedule: lr_decay = 2.5% for lr > 0.0003
                                    lr_decay = 1.5% for 0.0003 >= lr > 0.0001
                                    lr_decay = 1% for 0.0001 >= lr

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
            lr_temp = 0.06  # * n_gpu TODO think about multi gpu lr

            for i in range(n_lr_decays):
                if lr_temp > 0.04:
                    lr_decay = 0.025  # standard for regression: 0.07, standard for PID: 0.02
                elif 0.04 >= lr_temp > 0.01:
                    lr_decay = 0.015  # standard for regression: 0.04, standard for PID: 0.01
                else:
                    lr_decay = 0.01  # standard for regression: 0.02, standard for PID: 0.005
                
                lr_temp = lr_temp * (1 - float(lr_decay))

            return lr_temp

    elif name == "triple_decay_reg":
        def learning_rate(n_epoch, n_file):
            """
            Function that calculates the current learning rate based on
            the number of already trained epochs.

            Learning rate schedule: lr_decay = 1.5% for lr > 0.0003
                                    lr_decay = 1.0% for 0.0003 >= lr > 0.0001
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
            lr_temp = 0.005  # * n_gpu TODO think about multi gpu lr

            for i in range(n_lr_decays):
                if lr_temp > 0.003:
                    lr_decay = 0.015  # standard for regression: 0.07, standard for PID: 0.02
                elif 0.003 >= lr_temp > 0.001:
                    lr_decay = 0.01  # standard for regression: 0.04, standard for PID: 0.01
                else:
                    lr_decay = 0.005  # standard for regression: 0.02, standard for PID: 0.005
                
                lr_temp = lr_temp * (1 - float(lr_decay))

            return lr_temp

    elif name == "triple_decay_cf":
        def learning_rate(n_epoch, n_file):
            """
            Function that calculates the current learning rate based on
            the number of already trained epochs.

            Learning rate schedule: lr_decay = 3% for lr > 0.015
                                    lr_decay = 2% for 0.015 >= lr > 0.01
                                    lr_decay = 1% for 0.0001 >= lr

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
            lr_temp = 0.025  # * n_gpu TODO think about multi gpu lr

            for i in range(n_lr_decays):
                if lr_temp > 0.015:
                    lr_decay = 0.03  # standard for regression: 0.07, standard for PID: 0.02
                elif 0.015 >= lr_temp > 0.01:
                    lr_decay = 0.02  # standard for regression: 0.04, standard for PID: 0.01
                else:
                    lr_decay = 0.01  # standard for regression: 0.02, standard for PID: 0.005
                
                lr_temp = lr_temp * (1 - float(lr_decay))

            return lr_temp


    else:
        raise NameError("Unknown orca learning rate name", name)

    return learning_rate
