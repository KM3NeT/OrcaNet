#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shuffles .h5 files. Can only be used for files where each dataset has the same number of rows (axis_0).
Be careful to not run out of memory! Needs the unshuffled .h5 file's disk space + the python overhead as memory."""

import sys
import os
import argparse
import numpy as np
import h5py
#from memory_profiler import profile # for memory profiling, call with @profile; myfunc()

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__ = '1.0'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Production'


def parse_input():
    """
    Parses the user input in order to return the most important information:
    1) list of files that should be shuffled 2) if the unshuffled file should be deleted 3) if the user wants to use chunks or not.
    :return: list file_list: list that contains all filepaths of the input files.
    :return: bool delete_flag: specifies if the old, unshuffled file should be deleted after extracting the data.
    :return: (bool, int) chunking: specifies if chunks should be used and if yes which size the chunks should have.
    :return (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving.
    """

    parser = argparse.ArgumentParser(description='E.g. < python shuffle_h5.py filepath_1 [filepath_2] [...] > \n'
                                                 'Shuffles .h5 files. Requires that each dataset of the files has the same number of rows (axis_0). \n'
                                                 'Outputs a new, shuffled .h5 file with the suffix < _shuffled >. \n'
                                                 'The old, unshuffled file can be deleted in the process with the optional argument --delete. \n'
                                                 'Additionally, chunking can be used for the shuffle output .h5 file by using --chunksize. \n'
                                                 'By default, chunking is disabled. \n'
                                                 'If you need fast I/O on the output file, you should consider to set the chunksize according to your use case.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('files', metavar='file', type=str, nargs='+', help = 'a file that should be shuffled, can be more than one argument.')
    parser.add_argument('-d', '--delete', action='store_true',
                        help = 'deletes the original input file after the shuffled .h5 is created.')
    parser.add_argument('-c', '--chunksize', dest='chunksize', type=int,
                        help = 'specify a chunksize value in order to use chunked storage for the shuffled .h5 file (default: not chunked).')
    parser.add_argument('-g', '--compression', dest='compression', action='store_true',
                        help = 'if a gzip filter with compression 1 should be used for saving. Only works with -c option!')
    parser.add_argument('-n', '--n_shuffles', dest='n_shuffles', type=int,
                        help = 'specify if multiple shuffles should be executed on a single file.') #TODO don't combine with -d option
    parser.set_defaults(compression=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    file_list = []
    for filepath in args.files:
        file_list.append(filepath)

    delete_flag = False
    if args.delete:
        delete_flag = True

    chunking = (False, None)
    if args.chunksize:
        chunking = (True, args.chunksize)

    compress=(None, None)
    if args.compression is True:
        compress = ('gzip', 1)

    n_shuffles = 1
    if args.n_shuffles:
        n_shuffles = args.n_shuffles

    return file_list, delete_flag, chunking, compress, n_shuffles


def shuffle_h5(filepath, delete_flag=True, chunking=(False, None), tool=False, compress=(None, None), n_shuffles=1):
    """
    Shuffles a .h5 file where each dataset needs to have the same number of rows (axis_0).
    The shuffled data is saved to a new .h5 file with the suffix < _shuffled.h5 >.
    :param str filepath: filepath of the unshuffled input file.
    :param bool delete_flag: specifies if the old, unshuffled file should be deleted after extracting the data.
    :param (bool, int) chunking: specifies if chunks should be used and if yes which size the chunks should have.
    :param bool tool: specifies if the function is accessed from the shuffle_h5_tool.
                      In this case, the shuffled .h5 file is returned instead of closed.
    :param (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving. ('gzip', 1)
    :return: h5py.File output_file_shuffled: returns the shuffled .h5 file object if it is called from the tool.
    """
    input_file = h5py.File(filepath, 'r')
    folder_data_array_dict = {}

    for folder_name in input_file:
        folder_data_array = input_file[folder_name][()] # get whole numpy array into memory
        folder_data_array_dict[folder_name] = folder_data_array # workaround in order to be able to close the input file at the next step

    input_file.close()
    if delete_flag is True: os.remove(filepath)

    filepath_without_extension = os.path.splitext(filepath)[0]
    output_file_shuffled = None
    for i in xrange(n_shuffles):

        if '_shuffled' in filepath_without_extension:
            # we don't want to create a file with hundreds of _shuffled suffixes
            output_file_shuffled = h5py.File(filepath_without_extension + '_' + str(i) + '.h5', 'w')
        else:
            output_file_shuffled = h5py.File(filepath_without_extension + '_shuffled_' + str(i) + '.h5', 'w')

        for n, dataset_key in enumerate(folder_data_array_dict):

            dataset = folder_data_array_dict[dataset_key]

            if n == 0:
                # get a particular seed for the first dataset such that the shuffling is consistent across the datasets
                r = np.random.RandomState()
                state = r.get_state()
                r.shuffle(dataset)

            else:
                r.set_state(state) # recover shuffle seed of the first dataset
                r.shuffle(dataset)

            chunks = (chunking[1],) + dataset.shape[1:] if chunking[0] is True else None
            output_file_shuffled.create_dataset(dataset_key, data=dataset, dtype=dataset.dtype, chunks=chunks,
                                                                compression=compress[0], compression_opts=compress[1])

        # don't close in the case of tool=True and the last iteration of the loop
        if tool is False or tool is True and i < n_shuffles-1: output_file_shuffled.close()

    if tool is True: return output_file_shuffled


def shuffle_h5_tool():
    """
    Frontend for the shuffle_h5 function that can be used in a bash environment.
    Shuffles .h5 files where each dataset needs to have the same number of rows (axis_0) for a single file.
    Saves the shuffled data to a new .h5 file.
    """
    file_list, delete_flag, chunking, compress, n_shuffles = parse_input()
    print n_shuffles

    for filepath in file_list:
        print 'Shuffling file ' + filepath
        output_file_shuffled = shuffle_h5(filepath, delete_flag=delete_flag, chunking=chunking, tool=True, compress=compress, n_shuffles=n_shuffles)

        print 'Finished shuffling. Output information:'
        print '---------------------------------------'
        print 'The output file contains the following datasets:'
        for dataset_name in output_file_shuffled:
            print 'Dataset ' + dataset_name + ' with the following shape, dtype and chunks (first argument is the chunksize in axis_0): \n' \
                  + str(output_file_shuffled[dataset_name].shape) + ' ; ' + str(output_file_shuffled[dataset_name].dtype) + ' ; ' + str(
                output_file_shuffled[dataset_name].chunks)

        output_file_shuffled.close()


if __name__ == '__main__':
    shuffle_h5_tool()
