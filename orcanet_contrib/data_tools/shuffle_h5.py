#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains functions to shuffles .h5 files.

Can only be used for files where each dataset has the same number of rows (axis_0).
A fixed random seed (42) is used for the shuffling!

Currently, two types of .h5 files are supported:

1) Files which can be read by km3pipe (e.g. files produced with OrcaSong).
2) Plain hdf5 files with a hdf5 folder depth of 1. This method is based on some legacy code.
   Be careful to not run out of memory! Needs the unshuffled .h5 file's disk space + the python overhead as memory.
   If you want to use it, please use the --legacy_mode option.
"""

import sys
import os
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import h5py
import km3pipe as kp
import km3modules as km

# from memory_profiler import profile # for memory profiling, call with @profile; myfunc()

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__email__ = 'michael.m.moser@fau.de'


def parse_input():
    """
    Parses the user input in order to return the most important information:

    1) list of files that should be shuffled
    2) if the unshuffled file should be deleted
    3) if the user wants to use a custom chunksize, or if the chunksize should be read from the input file.
    4) if the user wants to use a custom complib, or if the complib should be read from the input file.
    5) if the user wants to use a custom complevel, or if the complevel should be read from the input file.

    Returns
    -------
    input_files_list : list
        List that contains all filepaths of the input files that should be shuffled.
    delete : bool
        Boolean flag that specifies, if the unshuffled input files should be deleted after the shuffling.
    chunksize : None/int
        Specifies the chunksize for axis_0 in the shuffled output files.
        If None, the chunksize is read from the input files.
        Else, a custom chunksize will be used.
    complib : None/str
        Specifies the compression library that should be used for saving the shuffled output files.
        If None, the compression library is read from the input files.
        Else, a custom compression library will be used.
        Currently available: 'gzip', or 'lzf'.
    complevel : None/int
        Specifies the compression level that should be used for saving the shuffled output files.
        A compression level is only available for gzip compression, not lzf!
        If None, the compression level is read from the input files.
        Else, a custom compression level will be used.
    legacy_mode : bool
        Boolean flag that specifies, if the legacy shuffle mode should be used instead of the standard one.
        A more detailed description of this mode can be found in the summary at the top of this python file.

    """
    parser = ArgumentParser(description='E.g. < python shuffle_h5.py filepath_1 [filepath_2] [...] > \n'
                                        'Shuffles .h5 files. Requires that each dataset of the files has the same number of rows (axis_0). \n'
                                        'Outputs a new, shuffled .h5 file with the suffix < _shuffled >.',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('files', metavar='file', type=str, nargs='+', help='a .h5 file that should be shuffled, can be more than one argument.')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='deletes the original input file after the shuffled .h5 is created.')
    parser.add_argument('--chunksize', dest='chunksize', type=int,
                        help='Specify a chunksize value in order to use chunked storage for the shuffled .h5 file. \n'
                             ' Otherwise, it will be read from the input file..')
    parser.add_argument('--complib', dest='complib', type=str,
                        help='Specify a filter that should be used for compression. Either "gzip" or "lzf". \n'
                             'Otherwise, the filter will be read from the input file.')
    parser.add_argument('--complevel', dest='complevel', type=int,
                        help='Specify a compression filter strength that should be used for the compression. \n'
                             'Otherwise, the filter will be read from the input file. \n'
                             'Can range from 0 to 9. Has no effect on "lzf" compression.')
    parser.add_argument('--legacy_mode', dest='legacy_mode', action='store_true',
                        help='If you want to use the legacy mode, as described in the summary at the top of this python file.')

    parser.set_defaults(legacy_mode=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    input_files_list = []
    for filepath in args.files:
        input_files_list.append(filepath)

    delete = False
    if args.delete:
        delete = True
        print('You chose delete = True')

    chunksize = None
    if args.chunksize:
        chunksize = args.chunksize
        print('You chose chunksize = ' + str(chunksize))

    complib = None
    if args.complib:
        complib = args.complib
        print('You chose complib = ' + complib)

    complevel = None
    if args.complevel:
        complevel = args.complevel
        print('You chose complevel = ' + str(complevel))

    legacy_mode = args.legacy_mode

    return input_files_list, delete, chunksize, complib, complevel, legacy_mode


def get_f_compression_and_chunking(filepath):
    """
    Function that gets the used compression library, the compression level (if applicable)
    and the chunksize of axis_0 of the first dataset of the file.

    Parameters
    ----------
    filepath : str
        Filepath of a .hdf5 file.

    Returns
    -------
    compression : str
        The compression library that has been identified in the input file. E.g. 'gzip', or 'lzf'.
    complevel : int
        The compression level that has been identified in the input file.
    chunksize : None/int
        The chunksize of axis_0 that has been indentified in the input file.

    """
    f = h5py.File(filepath, 'r')

    # remove any keys to pytables folders that may be in the file
    f_keys_stripped = [x for x in list(f.keys()) if '_i_' not in x]

    compression = f[f_keys_stripped[0]].compression  # compression filter
    compression_opts = f[f_keys_stripped[0]].compression_opts  # filter strength
    chunksize = f[f_keys_stripped[0]].chunks[0]  # chunksize along axis_0 of the dataset

    return compression, compression_opts, chunksize


def shuffle_h5(filepath_input, tool=False, seed=42, delete=True, chunksize=None, complib=None, complevel=None, legacy_mode=False):
    """
    Shuffles a .h5 file where each dataset needs to have the same number of rows (axis_0).
    The shuffled data is saved to a new .h5 file with the suffix < _shuffled.h5 >.

    Parameters
    ----------
    filepath_input : str
        Filepath of the unshuffled input file.
    tool : bool
        Specifies if the function is accessed from the shuffle_h5_tool.
        In this case, the shuffled .h5 file is returned.
    seed : int
        Sets a fixed random seed for the shuffling.
    delete : bool
        Specifies if the old, unshuffled file should be deleted after extracting the data.
    chunksize : None/int
        Specifies the chunksize for axis_0 in the shuffled output files.
        If None, the chunksize is read from the input files.
        Else, a custom chunksize will be used.
    complib : None/str
        Specifies the compression library that should be used for saving the shuffled output files.
        If None, the compression library is read from the input files.
        Else, a custom compression library will be used.
        Currently available: 'gzip', or 'lzf'.
    complevel : None/int
        Specifies the compression level that should be used for saving the shuffled output files.
        A compression level is only available for gzip compression, not lzf!
        If None, the compression level is read from the input files.
        Else, a custom compression level will be used.
    legacy_mode : bool
        Boolean flag that specifies, if the legacy shuffle mode should be used instead of the standard one.
        A more detailed description of this mode can be found in the summary at the top of this python file.

    Returns
    -------
    output_file_shuffled : h5py.File
        H5py file instance of the shuffled output file.

    """
    complib_f, complevel_f, chunksize_f = get_f_compression_and_chunking(filepath_input)

    chunksize = chunksize_f if chunksize is None else chunksize
    complib = complib_f if complib is None else complib
    complevel = complevel_f if complevel is None else complevel

    if complib == 'lzf':
        complevel = None

    filepath_input_without_ext = os.path.splitext(filepath_input)[0]
    filepath_output = filepath_input_without_ext + '_shuffled.h5'

    if not legacy_mode:
        # set random km3pipe (=numpy) seed
        print('Setting a Global Random State with the seed < 42 >.')
        km.GlobalRandomState(seed=seed)

        # km3pipe uses pytables for saving the shuffled output file, which has the name 'zlib' for the 'gzip' filter
        if complib == 'gzip':
            complib = 'zlib'

        pipe = kp.Pipeline(timeit=True)  # add timeit=True argument for profiling
        pipe.attach(km.common.StatusBar, every=200)
        pipe.attach(km.common.MemoryObserver, every=200)
        pipe.attach(kp.io.hdf5.HDF5Pump, filename=filepath_input, shuffle=True, reset_index=True)
        pipe.attach(kp.io.hdf5.HDF5Sink, filename=filepath_output, complib=complib, complevel=complevel, chunksize=chunksize, flush_frequency=1000)
        pipe.drain()
        if delete:
            os.remove(filepath_input)

        output_file_filepath = filepath_output if delete is False else filepath_input
        output_file_shuffled = h5py.File(output_file_filepath, 'r+')

        # delete folders with '_i_' that are created by pytables in the HDF5Sink, we don't need them
        for folder_name in output_file_shuffled:
            if folder_name.startswith('_i_'):
                del output_file_shuffled[folder_name]

    else:
        input_file = h5py.File(filepath_input, 'r')
        folder_data_array_dict = {}

        for folder_name in input_file:
            folder_data_array = input_file[folder_name][()]  # get whole numpy array into memory
            folder_data_array_dict[folder_name] = folder_data_array  # workaround in order to be able to close the input file at the next step

        input_file.close()

        if delete:
            os.remove(filepath_input)

        output_file_shuffled = h5py.File(filepath_output, 'w')
        for n, dataset_key in enumerate(folder_data_array_dict):

            dataset = folder_data_array_dict[dataset_key]

            if n == 0:
                # get a particular seed for the first dataset such that the shuffling is consistent across the datasets
                r = np.random.RandomState(seed)
                state = r.get_state()
                r.shuffle(dataset)

            else:
                r.set_state(state)  # recover shuffle seed of the first dataset
                r.shuffle(dataset)

            chunks = (chunksize,) + dataset.shape[1:]
            output_file_shuffled.create_dataset(dataset_key, data=dataset, dtype=dataset.dtype, chunks=chunks,
                                                compression=complib, compression_opts=complevel)

    # close file in the case of tool=True
    if tool is False:
        output_file_shuffled.close()
    else:
        return output_file_shuffled


def shuffle_h5_tool():
    """
    Frontend for the shuffle_h5 function that can be used in a bash environment.

    Shuffles .h5 files where each dataset needs to have the same number of rows (axis_0) for a single file.
    Saves the shuffled data to a new .h5 file.
    """
    input_files_list, delete, chunksize, complib, complevel, legacy_mode = parse_input()

    for filepath_input in input_files_list:
        print('Shuffling file ' + filepath_input)
        output_file_shuffled = shuffle_h5(filepath_input, tool=True, seed=42, delete=delete, chunksize=chunksize,
                                          complib=complib, complevel=complevel, legacy_mode=legacy_mode)
        print('Finished shuffling. Output information:')
        print('---------------------------------------')
        print('The output file contains the following datasets:')
        for dataset_name in output_file_shuffled:
            print('Dataset ' + dataset_name + ' with the following shape, dtype and chunks '
                  '(first argument is the chunksize in axis_0): \n' + str(output_file_shuffled[dataset_name].shape)
                  + ' ; ' + str(output_file_shuffled[dataset_name].dtype) + ' ; '
                  + str(output_file_shuffled[dataset_name].chunks))

        output_file_shuffled.close()


if __name__ == '__main__':
    shuffle_h5_tool()
