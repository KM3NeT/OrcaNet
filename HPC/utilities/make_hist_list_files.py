#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Submits a job to concatenate the .h5 files in a specified folder.
"""

import os
import sys
import argparse
import numpy as np
import natsort as ns
import math


def parse_input():
    """
    Parses the user input in order to return the most important information:
    1) directory where the .h5 files that should be concatenated are located 2) the fraction of the data that should be used for the test split
    3) the total number of train files 4) the total number of test files
    :return: str dirpath: full path of the folder with the .h5 files.
    :return: float test_fraction: fraction of the total data that should be used for the test data sample.
    :return: int, int n_train_files, n_test_files: number of concatenated .h5 train/test files.
    :return int n_file_start: specifies the first file number of the .h5 files (standard: 1).
    :return None/int n_files_max: specifies the maximum file number upon which the concatenation should happen.
    :return: (bool, int) chunking: specifies if chunks should be used and if yes which size the chunks should have.
    :return (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving.
    """
    parser = argparse.ArgumentParser(description='Parses the user input in order to return the most important information:\n'
                                                 '1) directory where the .h5 files that should be concatenated are located\n '
                                                 '2) the fraction of the data that should be used for the test split\n'
                                                 '3) the total number of train files 4) the total number of test files',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('dirpath', metavar='dirpath', type=str, nargs=1,
                        help='the path where the .h5 files are located.')
    parser.add_argument('test_fraction', metavar='test_fraction', type=float, nargs=1,
                        help='the fraction of files that should be used for the test data sample.')
    parser.add_argument('n_train_files', metavar='n_train_files', type=int, nargs=1,
                        help='into how many files the train data sample should be split.')
    parser.add_argument('n_test_files', metavar='n_test_files', type=int, nargs=1,
                        help='into how many files the test data sample should be split.')

    parser.add_argument('--n_file_start', dest='n_file_start', type=int,
                        help='the file number of the first file (standard: 1).')
    parser.add_argument('--n_files_max', dest='n_files_max', type=int,
                        help='if you do not want to use ALL h5 files that are in the dirpath folder.')
    parser.add_argument('-g', '--compression', dest='compression', action='store_true',
                        help = 'if a gzip filter with compression 1 should be used for saving. Only works with -c option!')
    parser.add_argument('-c', '--chunksize', dest='chunksize', type=int,
                        help = 'specify a chunksize value in order to use chunked storage for the concatenated .h5 file (default: not chunked).')
    parser.add_argument('-p', '--cuts', dest='cuts', action='store_true',
                        help = 'if cuts should be applied for the to be concatenated h5 files. '
                               'The txt files with the cut informationa are specified in the load_event_selection() function of concatenate_h5.py.')
    parser.set_defaults(compression=False)
    parser.set_defaults(cuts=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    dirpath = args.dirpath[0]
    test_fraction = args.test_fraction[0]
    n_train_files = args.n_train_files[0]
    n_test_files = args.n_test_files[0]

    n_file_start = 1
    if args.n_file_start:
        n_file_start = args.n_file_start

    n_files_max = None
    if args.n_files_max:
        n_files_max = args.n_files_max

    chunking = None
    if args.chunksize: chunking = '--chunksize ' + str(args.chunksize)

    compress = None
    if args.compression: compress = '--compression'

    cuts = None
    if args.cuts: cuts = '--cuts'

    return dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, chunking, compress, cuts


def get_filepaths(dirpath):
    """
    Returns the filepaths of all .h5 files that are located in a specific directory.
    :param str dirpath: path of the directory where the .h5 files are located.
    :return: list filepaths: list with the full filepaths of all .h5 files in the dirpath folder.
    """
    filepaths = []
    for f in os.listdir(dirpath):
        if f.endswith(".h5"):
            filepaths.append(f)

    filepaths = ns.natsorted(filepaths)
    return filepaths


def get_f_property_indices(filepaths):
    """
    Returns the index of the file number and the index of the projection type of .h5 files based on the filenames in <filepaths>.
    :param list filepaths: list with the full filepaths of various .h5 files.
    :return: int index_f_number: index of the file number in a filepath str.
    :return: int index_proj_type: index of the projection type in a filepath str.
    """
    # find index of file number, which is defined to be following the '2016' identifier
    identifier_year = '2016'
    index_f_number = filepaths[0].split('_').index(identifier_year) + 1 # f_number is one after year information
    index_proj_type = index_f_number + 1

    return index_f_number, index_proj_type


def get_f_properties(filepaths, index_proj_type):
    """
    Returns the particle type and projection type information for the .h5 filepaths.
    :param list filepaths: list with the full filepaths of various .h5 files.
    :param int index_proj_type: index of the projection type in a filepath str.
    :return: str ptype_str: string with particle type information, e.g. 'elec-CC_and_muon-CC' if strings with both type are given.
    :return: str proj_type: projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    """
    # get all particle_types
    particle_types = []
    for f in filepaths:
        p_type = f.split('_')[3]  # index for particle_type in filename
        if p_type not in particle_types:
            particle_types.append(p_type)

    # make ptype string for the savenames of the .list files
    ptype_str = ''
    for i in xrange(len(particle_types)):
        if i == 0:
            ptype_str += particle_types[0]
        else:
            ptype_str += '_and_' + particle_types[i]

    # get projection_type
    proj_type = filepaths[0].split('_')[index_proj_type].translate(None, '.h5')  # strip .h5 from string

    return ptype_str, proj_type


def save_filepaths_to_list(dirpath, filepaths, include_range, p_type, proj_type, index_f_number, sample_type=''):
    """
    Saves the filepaths in the list <filepaths> to a .list file.
    The variable 'include_range' can be used to only save filepaths with file numbers in between a certain range.
    :param str dirpath: the directory where the .list file should be saved.
    :param list filepaths: list with the full filepaths of various .h5 files.
    :param ndarray(dim=1) include_range: 1D array that specifies into how many files the filepaths should be split.
                                         E.g. [0, 120, 240, 360, 480] for 4 total list files: 1-120, 121-240, 241-360, 361-480.
    :param str p_type: string with particle type information for the name of the .list file, e.g. 'elec-CC_and_muon-CC'.
    :param str proj_type: string with the projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    :param int index_f_number: index for the file number in a .split('_') list.
    :param str sample_type: additional information in the filename of the .list file with specifies if train/test.
    :return: list list_savenames: list that contains the savenames of all .list files that have been created.
    """
    n_files = len(include_range) - 1
    if sample_type != '': sample_type = sample_type + '_'

    list_savenames = []
    for i in xrange(n_files):
        savename = p_type + '_' + proj_type + '_' + sample_type + str(include_range[i] + 1) + '_to_' + str(include_range[i + 1]) + '.list'
        savepath = dirpath + '/' + savename
        list_savenames.append(savename)

        with open(savepath, 'w') as f_out:
            for f in filepaths:
                f_number = int(f.split('_')[index_f_number])
                if include_range[i] < f_number <= include_range[i+1]:
                    f_out.write(dirpath + '/' + f + '\n')

    return list_savenames


def user_input_sanity_check(n_test_files, test_fraction, n_test_start, n_test_end,
                            n_train_files, n_train_start_minus_one, n_train_end,
                            n_total_files):
    """
    Sanity check in order to verify that the user doesn't input data splits that are not possible.
    [I.e., we can only split into an integer amount of files.]
    """
    n_train_split = n_total_files * float(test_fraction)
    if not n_train_split.is_integer():
        raise ValueError(str(n_total_files) +
                        ' cannot be split in whole numbers with a test fraction of ' + str(test_fraction))

    range_validation = np.linspace(n_test_start, n_test_end, n_test_files + 1)
    for step in range_validation:
        if not step.is_integer():
            raise ValueError('The test data cannot be split equally with ' + str(n_test_files) + ' test files.')

    range_train = np.linspace(n_train_start_minus_one, n_train_end, n_test_files + 1)
    for step in range_train:
        if not step.is_integer():
            raise ValueError('The train data cannot be split equally with ' + str(n_train_files) + ' train files.')


def make_list_files(dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max):
    """
    Makes .list files of .h5 files in a <dirpath> based on a certain test_fraction and a specified number of train/test files.
    :param str dirpath: path of the directory where the .h5 files are located.
    :param float test_fraction: fraction of the total data that should be used for the test data sample.
    :param int n_train_files: number of concatenated .h5 train files.
    :param int n_test_files: number of concatenated .h5 test files.
    :param int n_file_start: specifies the first file number of the .h5 files (standard: 1).
    :param None/int n_files_max: specifies the maximum file number upon which the concatenation should happen.
    :return: list savenames_tt: list that contains the savenames of all created .list files.
    :return str p_type: string with particle type information for the name of the .list file, e.g. 'elec-CC_and_muon-CC'.
    :return str proj_type: string with the projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    """
    filepaths = get_filepaths(dirpath)

    # find index of file number, which is defined to be following the '2016' identifier
    index_f_number, index_proj_type = get_f_property_indices(filepaths)
    p_type, proj_type = get_f_properties(filepaths, index_proj_type)

    # get total number of files
    n_total_files = int(max(int(i.split('_')[index_f_number]) for i in filepaths)) if n_files_max is None else n_files_max
    range_all = np.linspace(0, n_total_files, 2, dtype=np.int) # range for ALL files

    # test
    n_test_start = (n_total_files - test_fraction * n_total_files)
    n_test_end = n_total_files
    range_test = np.linspace(n_test_start, n_test_end, n_test_files+1, dtype=np.int)

    # train
    n_train_start_minus_one = n_file_start - 1
    n_train_end = (n_total_files - test_fraction * n_total_files)

    range_train = np.linspace(n_train_start_minus_one, n_train_end, n_train_files+1, dtype=np.int)  # [1,2,3,4,5....,480]

    user_input_sanity_check(n_test_files, test_fraction, n_test_start, n_test_end,
                            n_train_files, n_train_start_minus_one, n_train_end,
                            n_total_files)

    save_filepaths_to_list(dirpath, filepaths, range_all, p_type, proj_type, index_f_number)
    savenames_test = save_filepaths_to_list(dirpath, filepaths, range_test, p_type, proj_type, index_f_number, sample_type='test')
    savenames_train = save_filepaths_to_list(dirpath, filepaths, range_train, p_type, proj_type, index_f_number, sample_type='train')

    savenames_tt = savenames_train + savenames_test

    return savenames_tt, p_type, proj_type


def make_list_files_and_concatenate():
    """
    Wrapper function that does the following:
    1) get user input from parse_input()
    2) make all .list files which contains the filepaths of the .h5 files that should be concatenated
    3) make a submit script to concatenate the .h5 files in the .list files.
    """
    dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, chunking, compress, cuts = parse_input()

    savenames_tt, p_type, proj_type = make_list_files(dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max)
    submit_concatenate_list_files(savenames_tt, dirpath, p_type, proj_type, chunking, compress, cuts)


def submit_concatenate_list_files(savenames, dirpath, p_type, proj_type, chunking, compress, cuts):
    """
    Function that writes a qsub .sh files which concatenates all files inside the .list files from the <savenames> list.
    :param list savenames: list that contains the savenames of all created .list files.
    :param str dirpath: path of the directory where the .h5 files are located.
    :param str p_type: string with particle type information for the name of the .list file, e.g. 'elec-CC_and_muon-CC'.
    :param str proj_type: string with the projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    """
    if not os.path.exists(dirpath + '/logs/cout'): # check if /logs/cout folder exists, if not create it.
        os.makedirs(dirpath + '/logs/cout')
    if not os.path.exists(dirpath + '/concatenated/logs'): # check if /concatenated/logs folder exists, if not create it.
        os.makedirs(dirpath + '/concatenated/logs')

    xstr = lambda s: '' if s is None else str(s) # Make str(None) = ''

    # make qsub .sh file
    with open(dirpath + '/submit_concatenate_h5_' + p_type + '_' + proj_type + '.sh', 'w') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write('#\n')
        f.write('#PBS -o /home/woody/capn/mppi033h/logs/submit_concatenate_h5_${PBS_JOBID}.out -e /home/woody/capn/mppi033h/logs/submit_concatenate_h5_${PBS_JOBID}.err\n')
        f.write('\n')
        f.write('CodeFolder="/home/woody/capn/mppi033h/Code/HPC/cnns/utilities/data_tools"\n')
        f.write('cd ${CodeFolder}\n')
        f.write('chunksize="' + xstr(chunking) + '"\n')
        f.write('compression="' + xstr(compress) + '"\n')
        f.write('cuts="' + xstr(cuts) + '"\n')
        f.write('projection_path="' + dirpath + '"\n')
        f.write('\n')
        f.write('# lists with files that should be concatenated\n')

        n_cores = 4
        n_lists = len(savenames)
        n_loops = int(math.ceil(n_lists/float(n_cores)))

        for i in xrange(n_loops):
            n_lists_left = n_lists - i*n_cores
            write_txt_concatenate_files_one_loop(f, savenames, i, n_lists_left, n_cores)

    #os.system('qsub -l nodes=1:ppn=4,walltime=01:01:00 ' + dirpath + '/submit_concatenate_h5_' + p_type + '_' + proj_type + '.sh')


def write_txt_concatenate_files_one_loop(f, savenames, i, n_lists_left, n_cores):
    """
    Writes one n_cores loop which calls concatenate_h5.py for the .h5 files in the .list files.
    One iteration (loop) concatenates n_cores .list files.
    :param f: the qsub .sh file.
    :param list savenames: list that contains the savenames of all created .list files.
    :param int i: specifies how many times we have already looped over n_cores.
    :param int n_lists_left: remaining number of .lists of which the .h5 files still need to be concatenated.
    :param int n_cores: Number of cores and processes that should be used in one concatenation loop (standard: 4).
    """
    j = i*n_cores

    # define input (.list) and output filenames (concatenated .h5 files)
    loop_index = 0
    for n in range(j, j + n_cores): # write input and output filenames
        loop_index += 1

        if loop_index <= n_lists_left:
            name = os.path.splitext(savenames[n])[0] # get name without extension
            f.write('input_list_name_' + str(n) + '=' + name + '.list\n') # input
            f.write('output_list_name_' + str(n) + '=' + name + '.h5\n') # output

    # run concatenate script
    loop_index = 0
    for n in range(j, j + n_cores):
        loop_index += 1

        if loop_index <= n_lists_left:
            and_char = '&' if loop_index < n_lists_left else '' # in order to make execution on multiple cpus at the same time possible #TODO test
            f.write('(time taskset -c ' +  str(loop_index-1) + ' python concatenate_h5.py --list ${projection_path}/${input_list_name_' + str(n) + '} '
                    '${compression} ${cuts} ${chunksize} ${projection_path}/concatenated/${output_list_name_' + str(n) + '} > '
                    '${projection_path}/logs/cout/${output_list_name_' + str(n) + '}.txt) ' + and_char + '\n')

    f.write('wait\n')
    f.write('\n')


if __name__ == '__main__':
    make_list_files_and_concatenate()


# python make_hist_list_files.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/with_run_id/without_mc_time_fix/h5/xyzt 0.2 1 1 --compression --cuts --chunksize 32
# submit with e.g. 'qsub -l nodes=1:ppn=4,walltime=02:00:00 submit_concatenate_h5_elec-CC_and_muon-CC_xyzt.sh'




