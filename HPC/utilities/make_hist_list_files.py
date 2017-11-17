import os
import sys
import argparse
import numpy as np
import natsort as ns


def parse_input():
    """
        Parses the user input in order to return the most important information:
        1) list of files that should be shuffled 2) if the unshuffled file should be deleted 3) if the user wants to use chunks or not.
        :return: list file_list: list that contains all filepaths of the input files.
        :return: bool delete_flag: specifies if the old, unshuffled file should be deleted after extracting the data.
        :return: (bool, int) chunking: specifies if chunks should be used and if yes which size the chunks should have.
        :return (None/str, None/int) compress: Tuple that specifies if a compression should be used for saving.
        """

    parser = argparse.ArgumentParser(description='Placeholder',
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
    parser.set_defaults(compression=False)

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

    chunking = (False, None)
    if args.chunksize:
        chunking = (True, args.chunksize)

    compress = (None, None)
    if args.compression is True:
        compress = ('gzip', 1)


    return dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, chunking, compress


def get_filepaths(dirpath):
    filepaths = []
    for f in os.listdir(dirpath):
        if f.endswith(".h5"):
            filepaths.append(f)

    filepaths = ns.natsorted(filepaths)
    return filepaths

def get_f_property_indices(filepaths):

    # find index of file number, which is defined to be following the '2016' identifier
    identifier_year = '2016'
    index_f_number = filepaths[0].split('_').index(identifier_year) + 1 # f_number is one after year information
    index_proj_type = index_f_number + 1

    return index_f_number, index_proj_type

def get_f_properties(filepaths, index_proj_type):

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

    n_files = len(include_range) - 1
    if sample_type != '': sample_type = sample_type + '_'

    for i in xrange(n_files):
        savepath = dirpath + '/' + p_type + '_' + proj_type + '_' + sample_type + str(include_range[i] + 1) + '_to_' + str(include_range[i + 1]) + '.list'

        with open(savepath, 'w') as f_out:
            for f in filepaths:
                f_number = int(f.split('_')[index_f_number])
                if include_range[i] < f_number <= include_range[i+1]:
                    f_out.write(dirpath + '/' + f + '\n')

    # TODO store name of saved .list files for concatenating


def user_input_sanity_check(n_test_files, test_fraction, n_test_start_minus_one, n_test_end,
                            n_train_files, n_train_start_minus_one, n_train_end,
                            n_total_files):

    n_train_split = n_total_files * float(test_fraction)
    if not n_train_split.is_integer():
        raise ValueError(str(n_total_files) +
                        ' cannot be split in whole numbers with a test fraction of ' + str(test_fraction))

    range_validation = np.linspace(n_test_start_minus_one, n_test_end, n_test_files + 1)
    for step in range_validation:
        if not step.is_integer():
            raise ValueError('The test data cannot be split equally with ' + str(n_test_files) + ' test files.')

    range_train = np.linspace(n_train_start_minus_one, n_train_end, n_test_files + 1)
    for step in range_train:
        if not step.is_integer():
            raise ValueError('The train data cannot be split equally with ' + str(n_train_files) + ' train files.')


def make_list_files(dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max):

    filepaths = get_filepaths(dirpath)

    # find index of file number, which is defined to be following the '2016' identifier
    index_f_number, index_proj_type = get_f_property_indices(filepaths)
    p_type, proj_type = get_f_properties(filepaths, index_proj_type)

    # get total number of files
    n_total_files = int(max(int(i.split('_')[index_f_number]) for i in filepaths)) if n_files_max is None else n_files_max
    #n_total_files = int(max(int(i.split('_')[index_f_number]) for i in filepaths))
    range_all = np.linspace(0, n_total_files, 2, dtype=np.int)

    n_val_start_minus_one = (n_total_files - test_fraction * n_total_files)  # since the files don't start from 0 but 1.....
    n_val_end = n_total_files
    range_validation = np.linspace(n_val_start_minus_one, n_val_end, n_test_files+1, dtype=np.int)

    # train
    n_train_start_minus_one = n_file_start -1
    n_train_end = (n_total_files - test_fraction * n_total_files)

    range_train = np.linspace(n_train_start_minus_one, n_train_end, n_test_files+1, dtype=np.int)  # [1,2,3,4,5....,480]

    user_input_sanity_check(n_test_files, test_fraction, n_val_start_minus_one, n_val_end,
                            n_train_files, n_train_start_minus_one, n_train_end,
                            n_total_files)

    save_filepaths_to_list(dirpath, filepaths, range_all, p_type, proj_type, index_f_number)
    save_filepaths_to_list(dirpath, filepaths, range_validation, p_type, proj_type, index_f_number, sample_type='test')
    save_filepaths_to_list(dirpath, filepaths, range_train, p_type, proj_type, index_f_number, sample_type='train')


def make_list_files_and_concatenate():
    dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max, chunking, compress = parse_input()

    make_list_files(dirpath, test_fraction, n_train_files, n_test_files, n_file_start, n_files_max)

    # submit concatenate files
if __name__ == '__main__':
    make_list_files_and_concatenate()


# python make_hist_list_files.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/with_run_id/h5/xyzt 0.2 1 1 --compression --chunksize 32




