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
    :return str dirpath: full path of the folder with the .h5 files.
    :return float test_fraction: fraction of the total data that should be used for the test data sample.
    :return int, int n_train_files, n_test_files: number of concatenated .h5 train/test files.
    :return int n_file_start: specifies the first file number of the .h5 files (standard: 1).
    :return None/int n_files_max: specifies the maximum file number upon which the concatenation should happen.
    :return (bool, int) chunking: specifies if chunks should be used and if yes which size the chunks should have.
    :return (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving.
    :return bool cuts: specifies if cuts should be used during the concatenation process step.
    :return bool tau_test_only: specifies if possible tau files in the dirpath should only be used for the test dataset.
    :return str ignore_interaction_type: specifies, if a certain interaction type should be ignored in getting the filepaths.
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
    parser.add_argument('-t', '--tau_test_only', dest='tau_test_only', action='store_true',
                        help = 'specifies if tau files should only be used for the test split.')
    parser.add_argument('-i', '--ignore_interaction_type', dest='ignore_interaction_type', type=str, nargs='+',
                        help = 'specifies if files of a certain flavour should be ignored for making the list files. \n'
                               'Possible options: tau-CC, muon-CC, elec-NC, elec-CC')
    parser.set_defaults(compression=False)
    parser.set_defaults(cuts=False)
    parser.set_defaults(tau_test_only=False)

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

    tau_test_only = None
    if args.tau_test_only: tau_test_only = True

    ignore_interaction_type = False
    if args.ignore_interaction_type: ignore_interaction_type = args.ignore_interaction_type

    return dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, chunking, compress, cuts, tau_test_only, ignore_interaction_type


def get_filepaths(dirpath, ignore_interaction_type):
    """
    Returns the filepaths of all .h5 files that are located in a specific directory.
    :param str dirpath: path of the directory where the .h5 files are located.
    :param str ignore_interaction_type: specifies, if a certain interaction type should be ignored in getting the filepaths.
    :return: list filepaths: list with the full filepaths of all .h5 files in the dirpath folder.
    """
    filepaths = []
    for f in os.listdir(dirpath):
        if f.endswith(".h5"):
            if ignore_interaction_type is not False:
                #if ignore_interaction_type not in f: filepaths.append(f)
                if any(ignore_type in f for ignore_type in ignore_interaction_type):
                    continue
                else:
                    filepaths.append(f)
            else:
                filepaths.append(f)

    filepaths = ns.natsorted(filepaths)
    return filepaths


def get_particle_types(filepaths):
    """
    Gets the information which particle types are included in the filepaths files.
    :param list filepaths: list with the full filepaths of various .h5 files.
    :return list particle_types: list that contains the single particle types as strings, e.g. ['muon-CC', 'elec-CC', ...].
    :return str ptype_str: string with particle type information, e.g. 'elec-CC_and_muon-CC' if strings with both type are given.
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

        if ptype_str == '':
            ptype_str += particle_types[i]
        else:
            ptype_str += '_and_' + particle_types[i]

    return particle_types, ptype_str


def split_filepaths_into_interaction_types(filepaths, particle_types):
    """
    Splits the filepath list into a dict of lists which contain the filepaths for each particle type.
    :param filepaths: list with the full filepaths of various .h5 files, can contain multiple particle types.
    :param list particle_types: list that contains the single particle types as strings.
    :return: dict(list) filepaths_per_interaction_type: dict that contains the filepaths for each particle type.
                                                        E.g. {'muon-CC': [filepaths for muon-CC files], ...}.
    """
    filepaths_per_interaction_type = {}
    for interaction_type in particle_types:
        fpath_temp_list = []
        for fpath in filepaths:
            if interaction_type in fpath: fpath_temp_list.append(fpath)
        filepaths_per_interaction_type[interaction_type] = fpath_temp_list

    return filepaths_per_interaction_type


def get_f_properties(filepaths_per_interaction_type):
    """
    Gets various file properties from the files in the filepaths_per_interaction_type dict.
    :param dict(list) filepaths_per_interaction_type: dict that contains the filepaths for each particle type.
    :return dict index_f_number: contains the index of the file number in a filepath for each interaction type.
    :return str proj_type: a string from the filepaths that contains the projection type. Same for all interaction types!
    """
    # find index of file number for each interaction_type, which is defined to be following the '2016' identifier
    identifier_year = '2016'
    index_f_number, index_proj_type = {}, {}
    for interaction_type in filepaths_per_interaction_type:
        index_f_number[interaction_type] = filepaths_per_interaction_type[interaction_type][0].split('_').index(identifier_year) + 1 # f_number is one after year information
        index_proj_type[interaction_type] = index_f_number[interaction_type] + 1

    # get projection_type, should be same for ALL interaction types, take first one
    first_key = next(iter(filepaths_per_interaction_type))
    proj_type = filepaths_per_interaction_type[first_key][0].split('_')[index_proj_type[first_key]].translate(None, '.h5')  # strip .h5 from string

    return index_f_number, proj_type


def get_file_split_info_per_interaction_channel(filepaths_per_interaction_type, index_f_number, n_train_files, n_test_files, n_files_max, n_file_start, test_fraction, tau_test_only):
    """
    Function that creates the file splits into the (multiple) train/test files.
    :param dict(list) filepaths_per_interaction_type: dict that contains the filepaths for each particle type.
    :param dict index_f_number: contains the index of the file number in a filepath for each interaction type.
    :param int n_train_files: specifies into how many files the train dataset should be split.
    :param int n_test_files: specifies into how many files the test dataset should be split.
    :param None/int n_files_max: specifies the maximum number of files that should be used for the whole dataset (train+test).
                                 Same for all interaction types!
    :param int n_file_start: specifies the first file number of the .h5 files (standard: 1).
    :param float test_fraction: fraction of the total data that should be used for the test data sample.
    :param bool tau_test_only: specifies if possible tau files in the dirpath should only be used for the test dataset.
    :return: Various dataset split variables.
    """
    n_total_files, range_all = {}, {}
    n_test_start_minus_one, n_test_end, range_test = {}, {}, {}
    n_train_start, n_train_end, range_train = {}, {}, {}

    for interaction_type in filepaths_per_interaction_type:
        n_total_files[interaction_type] = int(max(int(i.split('_')[index_f_number[interaction_type]]) for i in filepaths_per_interaction_type[interaction_type])) if n_files_max is None else n_files_max
        range_all[interaction_type] = np.linspace(0, n_total_files[interaction_type], 2, dtype=np.int)

        # Test, use first files for test split
        n_test_start_minus_one[interaction_type] = n_file_start - 1
        n_test_end[interaction_type] = (n_total_files[interaction_type] - (1 - test_fraction) * n_total_files[interaction_type])
        range_test[interaction_type] = np.linspace(n_test_start_minus_one[interaction_type], n_test_end[interaction_type],
                                                   n_test_files + 1, dtype=np.int)

        # Train, use the rest of the files
        n_train_start[interaction_type] = (n_total_files[interaction_type] - (1 - test_fraction) * n_total_files[interaction_type])
        n_train_end[interaction_type] = n_total_files[interaction_type]
        range_train[interaction_type] = np.linspace(n_train_start[interaction_type], n_train_end[interaction_type], n_train_files + 1, dtype=np.int)  # [1,2,3,4,5....,480]

        if interaction_type == 'tau-CC' and tau_test_only is True:
            # fix tau-CC to be only included in the test split
            n_test_end[interaction_type] = n_total_files[interaction_type]
            range_test[interaction_type] = np.linspace(n_test_start_minus_one[interaction_type], n_test_end[interaction_type],
                                                   n_test_files + 1, dtype=np.int)
            # make it such that tau-CC is not included in the train split, only need to modify range_train['tau-CC']
            # will not pass 'if range[interaction_type][i] < f_number <= range[interaction_type][i+1]:'
            range_train[interaction_type] = [-1] * (n_train_files + 1)

    return n_total_files, range_all, n_test_start_minus_one, n_test_end, range_test, n_train_start, n_train_end, range_train


def save_filepaths_to_list(dirpath, filepaths_per_interaction_type, include_range, p_type_str, proj_type, index_f_number, tau_test_only, sample_type=''):
    """
    Saves the filepaths in the list <filepaths> to a .list file.
    The variable 'include_range' can be used to only save filepaths with file numbers in between a certain range.
    :param str dirpath: the directory where the .list file should be saved.
    :param dict(list) filepaths_per_interaction_type: dict that contains the filepaths for each particle type.
    :param ndarray(dim=1) include_range: 1D array that specifies into how many files the filepaths should be split.
                                         E.g. [0, 120, 240, 360, 480] for 4 total list files: 1-120, 121-240, 241-360, 361-480.
    :param str p_type_str: string with particle type information for the name of the .list file, e.g. 'elec-CC_and_muon-CC'.
    :param str proj_type: string with the projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    :param dict index_f_number: index for the file number in a .split('_') list. dict(str: int)
    :param str sample_type: additional information in the filename of the .list file with specifies if train/test.
    :return: list list_savenames: list that contains the savenames of all .list files that have been created.
    """
    if sample_type != '': sample_type = sample_type + '_'

    n_files = len(include_range[next(iter(include_range))]) - 1 # n_files is same for all interaction types, calculate based on first interaction type
    list_savenames = []
    for i in xrange(n_files):

        if sample_type=='train_' and tau_test_only is True:
            print p_type_str
            p_type_str = p_type_str.replace('_and_tau-CC', '')

        savename = p_type_str + '_' + proj_type + '_' + sample_type + 'file_' + str(i) + '.list'
        savepath = dirpath + '/' + savename
        list_savenames.append(savename)

        with open(savepath, 'w') as f_out:
            for interaction_type in filepaths_per_interaction_type:
                for f in filepaths_per_interaction_type[interaction_type]:
                    f_number = int(f.split('_')[index_f_number[interaction_type]])
                    if include_range[interaction_type][i] < f_number <= include_range[interaction_type][i+1]:
                        f_out.write(dirpath + '/' + f + '\n')

    return list_savenames


def user_input_sanity_check(n_test_files, test_fraction, n_test_start_minus_one, n_test_end,
                            n_train_files, n_train_start, n_train_end,
                            n_total_files):
    """
    Sanity check in order to verify that the user doesn't input data splits that are not possible.
    [I.e., we can only split into an integer amount of files.]
    """
    for interaction_type in n_total_files:
        n_train_split = n_total_files[interaction_type] * float(test_fraction)
        if not n_train_split.is_integer():
            raise ValueError(str(n_total_files) +
                            ' cannot be split in whole numbers with a test fraction of ' + str(test_fraction))

        range_validation = np.linspace(n_test_start_minus_one[interaction_type], n_test_end[interaction_type], n_test_files + 1)
        for step in range_validation:
            if not step.is_integer():
                raise ValueError('The test data cannot be split equally with ' + str(n_test_files) + ' test files.')

        range_train = np.linspace(n_train_start[interaction_type], n_train_end[interaction_type], n_test_files + 1)
        for step in range_train:
            if not step.is_integer():
                raise ValueError('The train data cannot be split equally with ' + str(n_train_files) + ' train files.')


def make_list_files(dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, tau_test_only, ignore_interaction_type):
    """
    Makes .list files of .h5 files in a <dirpath> based on a certain test_fraction and a specified number of train/test files.
    :param str dirpath: path of the directory where the .h5 files are located.
    :param float test_fraction: fraction of the total data that should be used for the test data sample.
    :param int n_train_files: number of concatenated .h5 train files.
    :param int n_test_files: number of concatenated .h5 test files.
    :param int n_file_start: specifies the first file number of the .h5 files (standard: 1).
    :param None/int n_files_max: specifies the maximum file number upon which the concatenation should happen.
    :param bool tau_test_only: specifies if possible tau files in the dirpath should only be used for the test dataset.
    :param str ignore_interaction_type: specifies, if a certain interaction type should be ignored in getting the filepaths.
    :return: list savenames_tt: list that contains the savenames of all created .list files.
    :return str p_type: string with particle type information for the name of the .list file, e.g. 'elec-CC_and_muon-CC'.
    :return str proj_type: string with the projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    """
    filepaths = get_filepaths(dirpath, ignore_interaction_type) # contains all filepaths of the files in the dirpath

    particle_types, p_type_str = get_particle_types(filepaths)
    filepaths_per_interaction_type = split_filepaths_into_interaction_types(filepaths, particle_types)
    index_f_number, proj_type = get_f_properties(filepaths_per_interaction_type)

    n_total_files, range_all, n_test_start_minus_one, n_test_end, range_test, n_train_start, n_train_end, range_train = get_file_split_info_per_interaction_channel(filepaths_per_interaction_type, index_f_number, n_train_files, n_test_files, n_files_max, n_file_start, test_fraction, tau_test_only)

    user_input_sanity_check(n_test_files, test_fraction, n_test_start_minus_one, n_test_end,
                            n_train_files, n_train_start, n_train_end, n_total_files)

    tau_ptype_str = '_and_tau-CC' if tau_test_only is True else ''
    savenames_test = save_filepaths_to_list(dirpath, filepaths_per_interaction_type, range_test, p_type_str + tau_ptype_str, proj_type, index_f_number, tau_test_only, sample_type='test')
    savenames_train = save_filepaths_to_list(dirpath, filepaths_per_interaction_type, range_train, p_type_str, proj_type, index_f_number, tau_test_only, sample_type='train')

    savenames_tt = savenames_train + savenames_test

    return savenames_tt, p_type_str, proj_type


def make_list_files_and_concatenate():
    """
    Wrapper function that does the following:
    1) get user input from parse_input()
    2) make all .list files which contains the filepaths of the .h5 files that should be concatenated
    3) make a submit script to concatenate the .h5 files in the .list files.
    """
    dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, chunking, compress, cuts, tau_test_only, ignore_interaction_type = parse_input()

    savenames_tt, p_type_str, proj_type = make_list_files(dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, tau_test_only, ignore_interaction_type)
    submit_concatenate_list_files(savenames_tt, dirpath, p_type_str, proj_type, chunking, compress, cuts)


def submit_concatenate_list_files(savenames, dirpath, p_type_str, proj_type, chunking, compress, cuts):
    """
    Function that writes a qsub .sh files which concatenates all files inside the .list files from the <savenames> list.
    :param list savenames: list that contains the savenames of all created .list files.
    :param str dirpath: path of the directory where the .h5 files are located.
    :param str p_type_str: string with particle type information for the name of the .list file, e.g. 'elec-CC_and_muon-CC'.
    :param str proj_type: string with the projection type of the .h5 files in the filepath strings, e.g. 'xyzt'.
    :param (bool, int) chunking: specifies if chunks should be used and if yes which size the chunks should have.
    :param (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving.
    :param bool cuts: specifies if cuts should be used during the concatenation process step.
    """
    if not os.path.exists(dirpath + '/logs/cout'): # check if /logs/cout folder exists, if not create it.
        os.makedirs(dirpath + '/logs/cout')
    if not os.path.exists(dirpath + '/concatenated/logs'): # check if /concatenated/logs folder exists, if not create it.
        os.makedirs(dirpath + '/concatenated/logs')

    xstr = lambda s: '' if s is None else str(s) # Make str(None) = ''

    # make qsub .sh file
    with open(dirpath + '/submit_concatenate_h5_' + p_type_str + '_' + proj_type + '.sh', 'w') as f:
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
        f.write('source activate /home/hpc/capn/mppi033h/.virtualenv/h5_to_histo_env/\n')
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
            and_char = '&' if loop_index < n_lists_left else '' # in order to make execution on multiple cpus at the same time possible
            f.write('(time taskset -c ' +  str(loop_index-1) + ' python concatenate_h5.py --list ${projection_path}/${input_list_name_' + str(n) + '} '
                    '${compression} ${cuts} ${chunksize} ${projection_path}/concatenated/${output_list_name_' + str(n) + '})' + and_char + '\n')

    f.write('wait\n')
    f.write('\n')


if __name__ == '__main__':
    make_list_files_and_concatenate()


# python make_hist_list_files.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-250+500_geo-fix_60b 0.25 4 1 --compression --chunksize 32
# submit with e.g. 'qsub -l nodes=1:ppn=4,walltime=02:00:00 submit_concatenate_h5_elec-CC_and_muon-CC_xyzt.sh'




