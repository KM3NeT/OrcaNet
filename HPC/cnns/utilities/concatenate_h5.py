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

def get_cum_number_of_rows(file_list):
    """
    Returns the cumulative number of rows (axis_0) in a list based on the specified input .h5 files.
    This information is needed for concatenating the .h5 files later on.
    Additionally, the average number of rows for all the input files is calculated, in order to derive a sensible chunksize (optimized for diskspace).
    :param list file_list: list that contains all filepaths of the input files.
    :return: list cum_number_of_rows_list: list that contains the cumulative number of rows (i.e. [0,100,200,300,...] if each file has 100 rows)
    :return: int mean_number_of_rows: specifies the average number of rows (rounded up to int) for the files in the file_list.
    """
    total_number_of_rows = 0
    cum_number_of_rows_list = [0]
    number_of_rows_list = [] # used for approximating the chunksize

    # Get total number of rows for the files in the list, faster than resizing the dataset in each iteration of the file loop in concatenate_h5_files()
    for file_name in file_list:
        f = h5py.File(file_name, 'r')
        # get number of rows from the first folder of the file -> each folder needs to have the same number of rows
        total_number_of_rows += f[f.keys()[0]].shape[0]
        cum_number_of_rows_list.append(total_number_of_rows)
        number_of_rows_list.append(f[f.keys()[0]].shape[0])

        f.close()

    mean_number_of_rows = math.ceil(np.mean(number_of_rows_list))
    return cum_number_of_rows_list, mean_number_of_rows


def parse_input():
    """
    Parses the user input in order to return the most important information:
    1) list of files that should be concatenated 2) the filepath of the output .h5 file 3) use custom chunksize or not
    :return: list file_list: list that contains all filepaths of the input files.
    :return: str output_filepath: specifies the filepath (path+name) of the output .h5 file.
    :return: (bool, int) custom_chunksize: specifies if a custom_chunksize should be used and if yes, which chunksize has been specified. I.e. (True, 1000).
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

    parser.add_argument('files', metavar='file', type=str, nargs='?', help = 'a file that should be concatenated, minimum of two.')
    parser.add_argument('output_filepath', metavar='output_filepath', type=str, nargs=1, help='filepath and name of the output .h5 file')
    parser.add_argument('-l', '--list', dest='list_file', type=str,
                        help = 'filepath of a .list file that contains all .h5 files that should be concatenated')
    parser.add_argument('-c', '--chunksize', dest='chunksize', type=int,
                        help = 'specify a specific chunksize that should be used instead of the automatic selection.')

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

    return file_list, output_filepath, custom_chunksize


def concatenate_h5_files():
    """
    Main code. Concatenates .h5 files with multiple datasets, where each dataset in one file needs to have the same number of rows (axis_0).
    Gets user input with aid of the parse_input() function. By default, the chunksize for the output .h5 file is automatically computed
    based on the average number of rows per file, in order to eliminate padding (wastes disk space).
    For faster I/O, the chunksize should be set by the user depending on the use case.
    In deep learning applications for example, the chunksize should be equal to the batch size that is used later on for reading the data.
    """

    file_list, output_filepath, custom_chunksize = parse_input()
    cum_rows_list, mean_number_of_rows = get_cum_number_of_rows(file_list)

    file_output = h5py.File(output_filepath, 'w')

    for n, input_file_name in enumerate(file_list):

        print 'Processing file ' + file_list[n]

        input_file = h5py.File(input_file_name, 'r')

        for folder_name in input_file:

            folder_data = input_file[folder_name]
            print 'Shape and dtype of dataset ' + folder_name + ': ' + str(folder_data.shape) + ' ; ' + str(folder_data.dtype)

            if n == 0:
                # first file; create the dummy dataset with no max shape
                maxshape = (None,) + folder_data.shape[1:] # change shape of axis zero to None

                if custom_chunksize[0] is True:
                    output_dataset = file_output.create_dataset(folder_name, data=folder_data, maxshape=maxshape,
                                                                chunks=(custom_chunksize[1],)+ folder_data.shape[1:])
                else:
                    output_dataset = file_output.create_dataset(folder_name, data=folder_data, maxshape=maxshape,
                                                                chunks=(mean_number_of_rows,)+ folder_data.shape[1:])
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
