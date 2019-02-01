#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Concatenates .h5 files. Works only for files where each dataset has the same number of rows."""

import h5py
import numpy as np
import math
from argparse import ArgumentParser, RawTextHelpFormatter
import sys
# from memory_profiler import profile # for memory profiling, call with @profile; myfunc()

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__ = '1.0'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Production'


def parse_input():
    """
    Parses the user input in order to return the most important information:

    1) list of files that should be concatenated
    2) the filepath of the output .h5 file
    3) use custom chunksize or not.

    Returns
    -------
    file_list : list
        List that contains all filepaths of the input files.
    output_filepath : str
        String that specifies the filepath (path+name) of the output .h5 file.
    chunksize : None/int
        Specifies the chunksize for axis_0 in the concatenated output files.
        If None, the chunksize is read from the first input file.
        Else, a custom chunksize will be used.
    complib : None/str
        Specifies the compression library that should be used for saving the concatenated output files.
        If None, the compression library is read from the first input file.
        Else, a custom compression library will be used.
        Currently available: 'gzip', or 'lzf'.
    complevel : None/int
        Specifies the compression level that should be used for saving the concatenated output files.
        A compression level is only available for gzip compression, not lzf!
        If None, the compression level is read from the first input file.
        Else, a custom compression level will be used.

    """
    parser = ArgumentParser(description='E.g. < python concatenate_h5.py file_1 file_2 /path/to/output.h5 > or '
                                        '< python concatenate_h5.py --list filepaths.txt /path/to/output.h5 >.\n'
                                        'Concatenates arrays stored in .h5 files for either multiple direct .h5 inputs or a .txt file of .h5 files (--list option).\n'
                                        'Outputs a new .h5 file with the concatenated arrays. This output is chunked!\n'
                                        'Careful: The folders of one file need to have the same number of rows (axis_0)!\n'
                                        'Make a .txt file with < find /path/to/files -name "file_x-*.h5" | sort --version-sort > listname.list >\n'
                                        'Chunksize: By default, the chunksize is set to the chunksize of the first inputfile!',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('files', metavar='file', type=str, nargs='*', help = 'a file that should be concatenated, minimum of two.')
    parser.add_argument('output_filepath', metavar='output_filepath', type=str, nargs=1, help='filepath and name of the output .h5 file')
    parser.add_argument('-l', '--list', dest='list_file', type=str,
                        help='filepath of a .list file that contains all .h5 files that should be concatenated')
    parser.add_argument('--chunksize', dest='chunksize', type=int,
                        help='Specify a chunksize value in order to use chunked storage for the concatenated .h5 file.'
                             ' Otherwise, it will be read from the first input file..')
    parser.add_argument('--complib', dest='complib', type=str,
                        help='Specify a filter that should be used for compression. Either "gzip" or "lzf". '
                             'Otherwise, the filter will be read from the first input file.')
    parser.add_argument('--complevel', dest='complevel', type=int,
                        help='Specify a compression filter strength that should be used for the compression. '
                             'Otherwise, the filter will be read from the first input file. '
                             'Can range from 0 to 9. Has no effect on "lzf" compression.')

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

    return file_list, output_filepath, chunksize, complib, complevel


def get_cum_number_of_rows(file_list):
    """
    Returns the cumulative number of rows (axis_0) in a list based on the specified input .h5 files.

    Parameters
    ----------
    file_list : list
        List that contains all filepaths of the input files.

    Returns
    -------
    cum_number_of_rows_list : list
        List that contains the cumulative number of rows (i.e. [0,100,200,300,...] if each file has 100 rows).

    """
    total_number_of_rows = 0
    cum_number_of_rows_list = [0]
    number_of_rows_list = []  # used for approximating the chunksize

    # Get total number of rows for the files in the list, faster than resizing the dataset in each iteration of the file loop in concatenate_h5_files()

    for file_name in file_list:
        f = h5py.File(file_name, 'r')

        # get number of rows from the first folder of the file -> each folder needs to have the same number of rows
        f_keys = list(f.keys())
        # remove pytables folders starting with '_i_', because the shape of its first axis does not correspond to the number of events in the file.
        # all other folders normally have an axis_0 shape that is equal to the number of events in the file.
        f_keys_stripped = [x for x in f_keys if '_i_' not in x]

        total_number_of_rows += f[f_keys_stripped[0]].shape[0]
        cum_number_of_rows_list.append(total_number_of_rows)
        number_of_rows_list.append(f[f_keys_stripped[0]].shape[0])

        f.close()

    return cum_number_of_rows_list


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


def concatenate_h5_files(output_filepath, file_list, cum_rows_list, chunksize, complib, complevel):
    """
    Function that concatenates hdf5 files based on an output_filepath and a file_list of input files.

    If the files contain group_info and x_indices folders (if the input files are coming from km3pipe output),
    the group-id / the index of the x_indices is fixed in order to not get duplicates of group-ids / x-indices.

    Parameters
    ----------
    output_filepath : str
        String that specifies the filepath (path+name) of the output .h5 file.
    file_list : list
        List that contains all filepaths of the input files.
    cum_rows_list : list
        List that contains the cumulative number of rows (i.e. [0,100,200,300,...] if each file has 100 rows).
    chunksize : None/int
        Specifies the chunksize for axis_0 in the concatenated output files.
        If None, the chunksize is read from the first input file.
        Else, a custom chunksize will be used.
    complib : None/str
        Specifies the compression library that should be used for saving the concatenated output files.
        If None, the compression library is read from the first input file.
        Else, a custom compression library will be used.
        Currently available: 'gzip', or 'lzf'.
    complevel : None/int
        Specifies the compression level that should be used for saving the concatenated output files.
        A compression level is only available for gzip compression, not lzf!
        If None, the compression level is read from the first input file.
        Else, a custom compression level will be used.

    """
    complib_f, complevel_f, chunksize_f = get_f_compression_and_chunking(file_list[0])

    chunksize = chunksize_f if chunksize is None else chunksize
    complib = complib_f if complib is None else complib
    complevel = complevel_f if complevel is None else complevel

    if complib == 'lzf':
        complevel = None

    file_output = h5py.File(output_filepath, 'w')

    for n, input_file_name in enumerate(file_list):
        print('Processing file ' + file_list[n])
        input_file = h5py.File(input_file_name, 'r')

        # create metadata
        if 'format_version' in list(input_file.attrs.keys()) and n == 0:
            file_output.attrs['format_version'] = input_file.attrs['format_version']

        for folder_name in input_file:

            if folder_name.startswith('_i_'):
                # we ignore datasets that have been created by pytables, don't need them anymore
                continue

            if n > 0 and folder_name in ['group_info', 'x_indices', 'y']:
                folder_data = input_file[folder_name][()]
                # we need to add the current number of the group_id / index in the file_output
                # to the group_ids / indices of the file that is to be appended
                column_name = 'group_id' if folder_name in ['group_info', 'y'] else 'index'
                # add 1 because the group_ids / indices start with 0
                folder_data[column_name] += np.amax(file_output[folder_name][column_name]) + 1

            else:
                folder_data = input_file[folder_name]

            print('Shape and dtype of dataset ' + folder_name + ': ' + str(folder_data.shape) + ' ; ' + str(folder_data.dtype))

            if n == 0:
                # first file; create the dummy dataset with no max shape
                maxshape = (None,) + folder_data.shape[1:]  # change shape of axis zero to None
                chunks = (chunksize,) + folder_data.shape[1:]

                output_dataset = file_output.create_dataset(folder_name, data=folder_data, maxshape=maxshape, chunks=chunks,
                                                            compression=complib, compression_opts=complevel)

                output_dataset.resize(cum_rows_list[-1], axis=0)

            else:
                file_output[folder_name][cum_rows_list[n]:cum_rows_list[n + 1]] = folder_data

        file_output.flush()

    print('Output information:')
    print('-------------------')
    print('The output file contains the following datasets:')
    for folder_name in file_output:
        print('Dataset ' + folder_name + ' with the following shape, dtype and chunks (first argument'
              ' is the chunksize in axis_0): \n' + str(file_output[folder_name].shape) + ' ; ' +
              str(file_output[folder_name].dtype) + ' ; ' + str(file_output[folder_name].chunks))

    file_output.close()


def main():
    """
    Main code. Concatenates .h5 files with multiple datasets, where each dataset in one file needs to have the same number of rows (axis_0).

    Gets user input with aid of the parse_input() function. By default, the chunksize for the output .h5 file is automatically computed.
    based on the average number of rows per file, in order to eliminate padding (wastes disk space).
    For faster I/O, the chunksize should be set by the user depending on the use case.
    In deep learning applications for example, the chunksize should be equal to the batch size that is used later on for reading the data.
    """
    file_list, output_filepath, chunksize, complib, complevel = parse_input()
    cum_rows_list = get_cum_number_of_rows(file_list)
    concatenate_h5_files(output_filepath, file_list, cum_rows_list, chunksize, complib, complevel)


if __name__ == '__main__':
    main()
