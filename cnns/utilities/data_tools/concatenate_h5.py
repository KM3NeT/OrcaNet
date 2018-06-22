#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Concatenates .h5 files. Works only for files where each dataset has the same number of rows."""

import h5py
import numpy as np
import math
import argparse
import sys
#from memory_profiler import profile # for memory profiling, call with @profile; myfunc()

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__ = '1.0'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Production'


def parse_input():
    """
    Parses the user input in order to return the most important information:
    1) list of files that should be concatenated 2) the filepath of the output .h5 file 3) use custom chunksize or not.
    :return: list file_list: list that contains all filepaths of the input files.
    :return: str output_filepath: specifies the filepath (path+name) of the output .h5 file.
    :return: (bool, int) custom_chunksize: specifies if a custom_chunksize should be used and if yes, which chunksize has been specified. I.e. (True, 1000).
    :return (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving. ('gzip', 1)
    :return bool cuts: Boolean that specifies, if cuts should be applied on the to be concatenated h5 files.
    """
    parser = argparse.ArgumentParser(description='E.g. < python concatenate_h5.py file_1 file_2 /path/to/output.h5 > or '
                                                 '< python concatenate_h5.py --list filepaths.txt /path/to/output.h5 >.\n'
                                                 'Concatenates arrays stored in .h5 files for either multiple direct .h5 inputs or a .txt file of .h5 files (--list option).\n'
                                                 'Outputs a new .h5 file with the concatenated arrays. This output is chunked!\n'
                                                 'Careful: The folders of one file need to have the same number of rows (axis_0)!\n'
                                                 'Make a .txt file with < find /path/to/files -name "file_x-*.h5" | sort --version-sort > listname.list >\n'
                                                 'Chunksize: By default, the chunksize is set as the average number of rows per file, in order to keep the filesize small.\n'
                                                 'If you need fast I/O on the output file, you should consider to set the chunksize according to your use case. \n'
                                                 'This can be done with the optional argument --chunksize.\n'
                                                 'I.e. if you feed the output data to a neural network in batches, you should set chunksize=batchsize.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('files', metavar='file', type=str, nargs='*', help = 'a file that should be concatenated, minimum of two.')
    parser.add_argument('output_filepath', metavar='output_filepath', type=str, nargs=1, help='filepath and name of the output .h5 file')
    parser.add_argument('-l', '--list', dest='list_file', type=str,
                        help = 'filepath of a .list file that contains all .h5 files that should be concatenated')
    parser.add_argument('-c', '--chunksize', dest='chunksize', type=int,
                        help = 'specify a specific chunksize that should be used instead of the automatic selection.')
    parser.add_argument('-g', '--compression', dest='compression', action='store_true',
                        help = 'if a gzip filter with compression 1 should be used for saving.')
    parser.add_argument('-p', '--cuts', dest='cuts', action='store_true',
                        help = 'if cuts should be applied for the to be concatenated h5 files. '
                               'The txt files with the cut informationa are specified in the load_event_selection() function.')
    parser.set_defaults(compression=False)
    parser.set_defaults(cuts=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.list_file:
        file_list = [line.rstrip('\n') for line in open(args.list_file)]
    else:
        file_list = []
        for filepath in args.files:
            file_list.append(filepath)

    output_filepath = args.output_filepath[0]

    custom_chunksize = (False, None)
    if args.chunksize:
        custom_chunksize = (True, args.chunksize)

    compress=(None, None)
    if args.compression is True:
        compress = ('gzip', 1)

    cuts = args.cuts

    return file_list, output_filepath, custom_chunksize, compress, cuts


def get_cum_number_of_rows(file_list, cuts=False):
    """
    Returns the cumulative number of rows (axis_0) in a list based on the specified input .h5 files.
    This information is needed for concatenating the .h5 files later on.
    Additionally, the average number of rows for all the input files is calculated, in order to derive a sensible chunksize (optimized for diskspace).
    :param list file_list: list that contains all filepaths of the input files.
    :param bool cuts: specifies if cuts should be used for getting the cum_number_of_rows.
                      In this case, the function also returns an events_survive_cut dict that specifies, if an event in a certain file survives the cut.
    :return: list cum_number_of_rows_list: list that contains the cumulative number of rows (i.e. [0,100,200,300,...] if each file has 100 rows).
    :return: int mean_number_of_rows: specifies the average number of rows (rounded up to int) for the files in the file_list.
    :return: None/dict dict_events_survive_cut: None if cuts=False, else it contains a dict with the information which events in a certain h5 file survive the cuts.
    """
    total_number_of_rows = 0
    cum_number_of_rows_list = [0]
    number_of_rows_list = []  # used for approximating the chunksize
    dict_events_survive_cut = None

    # Get total number of rows for the files in the list, faster than resizing the dataset in each iteration of the file loop in concatenate_h5_files()

    if cuts is True:
        dict_events_survive_cut = {}
        # get cum_number_of_rows with event cuts from a txt file
        cuts_run_id_event_id_pid = load_event_selection() # get run_id and event_id cuts (also needs particle_type and is_cc) from the txt file

        for file_name in file_list:
            f = h5py.File(file_name, 'r')
            event_info = f['y'][()] # load whole hdf5 dataset as numpy array
            event_info = event_info[:, [9,0,1,3]]  # use y dataset of the .h5 files that contain run_id, event_id, particle_type and is_cc
            event_in_cut_file = in_nd(event_info, cuts_run_id_event_id_pid, absolute=True)  # check which events in event_info are in the event cut array, returns boolean array

            n_events_survive_cut = np.count_nonzero(event_in_cut_file) # number of events that survive the cut
            total_number_of_rows += n_events_survive_cut
            cum_number_of_rows_list.append(total_number_of_rows)
            number_of_rows_list.append(n_events_survive_cut)

            dict_events_survive_cut[file_name] = event_in_cut_file # save event_in_cut_file information to dict for the concatenate_h5_files function

            f.close()

    else:
        for file_name in file_list:
            f = h5py.File(file_name, 'r')
            # get number of rows from the first folder of the file -> each folder needs to have the same number of rows
            total_number_of_rows += f[f.keys()[0]].shape[0]
            cum_number_of_rows_list.append(total_number_of_rows)
            number_of_rows_list.append(f[f.keys()[0]].shape[0])

            f.close()

    mean_number_of_rows = math.ceil(np.mean(number_of_rows_list))

    return cum_number_of_rows_list, mean_number_of_rows, dict_events_survive_cut


#-- Functions for applying event cuts to the to be concatenated files --#

def load_event_selection():
    """
    Loads the event & run id's that survive the cuts from a .txt file.
    After that, it adds a pid column to them and returns it.
    :return: ndarray(ndim=2) arr_sel_events: 2D array that contains [run_id, event_id, particle_type, is_cc]
                                                   for each event that survives the event cuts.
    """
    path = '/home/woody/capn/mppi033h/Code/HPC/cnns/results/plots/pheid_event_selection_txt/' # folder for storing the event cut .txts

    # Moritz's precuts
    particle_type_dict = {'muon-CC': ['muon_cc_3_100_selectedEvents_forMichael_01_18.txt', (14,1)],
                          'elec-CC': ['elec_cc_3_100_selectedEvents_forMichael_01_18.txt', (12,1)]}

    # Containment cut
    # particle_type_dict = {'muon-CC': ['muon_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt', (14,1)],
    #                       'elec-CC': ['elec_cc_3_100_selectedEvents_Rsmaller100_abszsmaller90_forMichael.txt', (12,1)]}

    arr_sel_events = None
    for key in particle_type_dict:
        txt_file = particle_type_dict[key][0]

        if arr_sel_events is None:
            arr_sel_events = np.loadtxt(path + txt_file, dtype=np.float32)
            arr_sel_events = add_pid_column_to_array(arr_sel_events, particle_type_dict, key)
        else:
            temp_pheid_sel_events = np.loadtxt(path + txt_file, dtype=np.float32)
            temp_pheid_sel_events = add_pid_column_to_array(temp_pheid_sel_events, particle_type_dict, key)

            arr_sel_events = np.concatenate((arr_sel_events, temp_pheid_sel_events), axis=0)

    return arr_sel_events # [run_id, event_id, particle_type, is_cc]


def add_pid_column_to_array(array, particle_type_dict, key):
    """
    Takes an array and adds two pid columns (particle_type, is_cc) to it along axis_1.
    :param ndarray(ndim=2) array: array to which the pid columns should be added.
    :param dict particle_type_dict: dict that contains the pid tuple (e.g. for muon-CC: (14,1)) for each interaction type at pos[1].
    :param str key: key of the dict that specifies which kind of pid tuple should be added to the array (dependent on interaction type).
    :return: ndarray(ndim=2) array_with_pid: array with additional pid columns. ordering: [array_columns, pid_columns]
    """
    # add pid columns particle_type, is_cc to events
    pid = np.array(particle_type_dict[key][1], dtype=np.float32).reshape((1,2))
    pid_array = np.repeat(pid, array.shape[0] , axis=0)

    array_with_pid = np.concatenate((array, pid_array), axis=1)
    return array_with_pid


def in_nd(a, b, absolute=True, assume_unique=False):
    """
    Function that generalizes the np in_1d function to nd.
    Checks if entries in axis_0 of a exist in b and returns the bool array for all rows.
    Kind of hacky by using str views on the np arrays.
    :param ndarray(ndim=2) a: array where it should be checked whether each row exists in b or not.
    :param ndarray(ndim=2) b: array upon which the rows of a are checked.
    :param bool absolute: Specifies if absolute() should be called on the arrays before applying in_nd.
                     Useful when e.g. in_nd shouldn't care about particle (+) or antiparticle (-).
    :param bool assume_unique: ff True, the input arrays are both assumed to be unique, which can speed up the calculation.
    :return: ndarray(ndim=1): Boolean array that specifies for each row of a if it also exists in b or not.
    """
    if a.dtype!=b.dtype: raise TypeError('The dtype of array a must be equal to the dtype of array b.')
    a, b = np.asarray(a, order='C'), np.asarray(b, order='C')

    if absolute is True: # we don't care about e.g. particles or antiparticles
        a, b = np.absolute(a), np.absolute(b)

    a = a.ravel().view((np.str, a.itemsize * a.shape[1]))
    b = b.ravel().view((np.str, b.itemsize * b.shape[1]))
    return np.in1d(a, b, assume_unique)


#-- Functions for applying event cuts to the to be concatenated files --#


def concatenate_h5_files():
    """
    Main code. Concatenates .h5 files with multiple datasets, where each dataset in one file needs to have the same number of rows (axis_0).
    Gets user input with aid of the parse_input() function. By default, the chunksize for the output .h5 file is automatically computed.
    based on the average number of rows per file, in order to eliminate padding (wastes disk space).
    For faster I/O, the chunksize should be set by the user depending on the use case.
    In deep learning applications for example, the chunksize should be equal to the batch size that is used later on for reading the data.
    """
    file_list, output_filepath, custom_chunksize, compress, cuts = parse_input()
    cum_rows_list, mean_number_of_rows, dict_events_survive_cut = get_cum_number_of_rows(file_list, cuts=cuts)

    file_output = h5py.File(output_filepath, 'w')

    for n, input_file_name in enumerate(file_list):

        print 'Processing file ' + file_list[n]

        input_file = h5py.File(input_file_name, 'r')

        for folder_name in input_file:

            folder_data = input_file[folder_name]
            if cuts is True:
                folder_data = folder_data[()] # load whole array into memory as np array
                folder_data = folder_data[dict_events_survive_cut[input_file_name]] # apply cuts to dataset

            print 'Shape and dtype of dataset ' + folder_name + ': ' + str(folder_data.shape) + ' ; ' + str(folder_data.dtype)

            if n == 0:
                # first file; create the dummy dataset with no max shape
                maxshape = (None,) + folder_data.shape[1:] # change shape of axis zero to None
                chunks = (custom_chunksize[1],) + folder_data.shape[1:] if custom_chunksize[0] is True else (mean_number_of_rows,) + folder_data.shape[1:]

                output_dataset = file_output.create_dataset(folder_name, data=folder_data, maxshape=maxshape, chunks=chunks,
                                                            compression=compress[0], compression_opts=compress[1])

                output_dataset.resize(cum_rows_list[-1], axis=0)

            else:
                file_output[folder_name][cum_rows_list[n]:cum_rows_list[n+1]] = folder_data

        file_output.flush()

    print 'Output information:'
    print '-------------------'
    print 'The output file contains the following datasets:'
    for folder_name in file_output:
        print 'Dataset ' + folder_name + ' with the following shape, dtype and chunks (first argument is the chunksize in axis_0): \n' \
              +  str(file_output[folder_name].shape) + ' ; ' + str(file_output[folder_name].dtype) + ' ; ' + str(file_output[folder_name].chunks)

    file_output.close()


if __name__ == '__main__':
    concatenate_h5_files()
