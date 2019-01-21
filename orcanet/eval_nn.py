#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main code for evaluating NN's.
It can also be called via a parser by running this python module as follows:

Usage:
    eval_nn.py FOLDER LIST CONFIG MODEL
    eval_nn.py (-h | --help)

Arguments:
    FOLDER  Path to the folder where everything gets saved to, e.g. the summary.txt, the plots, the trained models, etc.
    LIST    A .toml file which contains the pathes of the training and validation files.
            An example can be found in config/lists/example_list.toml
    CONFIG  A .toml file which sets up the training.
            An example can be found in config/models/example_config.toml. The possible parameters are listed in
            utilities/input_output_utilities.py in the class Settings.
    MODEL   Path to a .toml file with infos about a model.

Options:
    -h --help                       Show this screen.

"""

import matplotlib as mpl
from docopt import docopt
mpl.use('Agg')
from orcanet.utilities.nn_utilities import load_zero_center_data
from orcanet.utilities.evaluation_utilities import *
from orcanet.utilities.losses import get_all_loss_functions
from orcanet.utilities.input_output_utilities import Settings


# TODO Remove unnecessary input parameters to the following functions if they are already in the cfg
def predict_and_investigate_model_performance(cfg, model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean, arr_filename, folder_name):
    """
    Function that 1) makes predictions based on a Keras nn model and 2) investigates the performance of the model based on the predictions.

    Parameters
    ----------
    model : ks.model.Model
        Keras model of a neural network.
    test_files : list(([test_filepaths], test_filesize))
        List of tuples that contains the testfiles and their number of rows.
    n_bins : list(tuple(int))
        Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    batchsize : int
        Batchsize that is used for predicting.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname.
    modelname : str
        Name of the model.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/testdata.
    arr_filename : str
        Filename of the arr_nn_pred, which will be generated or loaded. It contains all predicitons of a model
        and true values, for a specific dataset.
    folder_name : str
        Name of the folder in the cnns directory in which everything will be saved.

    """
    # for layer in model.layers: # temp
    #     if 'batch_norm' in layer.name:
    #         layer.stateful = False

    if os.path.exists(arr_filename):
        print("Loading saved prediction:\n   "+arr_filename)
        arr_nn_pred = np.load(arr_filename)
    else:
        print("Generating new prediction.")
        arr_nn_pred = get_nn_predictions_and_mc_info(cfg, model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, samples=None)
        print("Done! Saving prediction as:\n   "+arr_filename)
        np.save(arr_filename, arr_nn_pred)

    # arr_nn_pred = np.load('results/plots/saved_predictions/arr_nn_pred_' + modelname + '_final_stateful_false.npy')
    # arr_nn_pred = np.load('results/plots/saved_predictions//arr_nn_pred_model_VGG_4d_xyz-t_and_yzt-x_and_4d_xyzt_track-shower_multi_input_single_train_tight-1_tight-2_lr_0.003_tr_st_test_st_final_stateful_false_1-100GeV_precut.npy')

    if class_type[1] == 'track-shower':  # categorical
        precuts = (False, '3-100_GeV_prod')

        make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title='Classified as track', filename=folder_name + 'plots/ts_' + modelname,
                                                      precuts=precuts, corr_cut_pred_0=0.5)

        make_prob_hists(arr_nn_pred, folder_name, modelname=modelname, precuts=precuts)
        make_hist_2d_property_vs_property(arr_nn_pred, folder_name, modelname, property_types=('bjorken-y', 'probability'),
                                          e_cut=(1, 100), precuts=precuts)
        calculate_and_plot_separation_pid(arr_nn_pred, folder_name, modelname, precuts=precuts)

    else:  # regression
        # TODO make the shallow reco not hardcoded
        arr_nn_pred_shallow = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')
        # precuts = (True, 'regr_3-100_GeV_prod_and_1-3_GeV_prod')
        precuts = (False, '3-100_GeV_prod')
        if 'energy' in class_type[1]:
            print('Generating plots for energy performance investigations')

            # DL
            make_2d_energy_resolution_plot(arr_nn_pred, modelname, folder_name, precuts=precuts,
                                           correct_energy=(True, 'median'))
            make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median_relative', precuts=precuts,
                                                      correct_energy=(True, 'median'), compare_shallow=(True, arr_nn_pred_shallow))
            make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, folder_name, precuts=precuts,
                                               compare_shallow=(True, arr_nn_pred_shallow), correct_energy=(True, 'median'))
            # shallow reco
            make_2d_energy_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)

        if 'dir' in class_type[1]:
            print('Generating plots for directional performance investigations')

            # DL
            make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', precuts=precuts,
                                              compare_shallow=(True, arr_nn_pred_shallow))
            make_2d_dir_correlation_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            # shallow reco
            make_2d_dir_correlation_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)

        if 'bjorken-y' in class_type[1]:
            print('Generating plots for bjorken-y performance investigations')

            # DL
            make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, folder_name, metric='median', precuts=precuts,
                                                    compare_shallow=(True, arr_nn_pred_shallow))
            make_2d_bjorken_y_resolution_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            # shallow reco
            make_2d_bjorken_y_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', folder_name, precuts=precuts)

        if 'errors' in class_type[1]:
            print('Generating plots for error performance investigations')

            make_1d_reco_err_div_by_std_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)  # TODO take precuts from above?
            make_1d_reco_err_to_reco_residual_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            make_2d_dir_correlation_plot_different_sigmas(arr_nn_pred, modelname, folder_name, precuts=precuts)


def get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident=''):
    """
    Derives the name of a model based on its number of bins and the class_type tuple.
    The final modelname is defined as 'model_Nd_proj_class_type[1]'.
    E.g. 'model_3d_xyz_muon-CC_to_elec-CC'.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) of the training images. Can contain multiple n_bins tuples.
    :param (int, str) class_type: Tuple that declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param str nn_arch: String that declares which neural network model architecture is used.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies the projection type.
    :param str str_ident: Optional str identifier that gets appended to the modelname.
    :return: str modelname: Derived modelname.
    """
    modelname = 'model_' + nn_arch + '_'

    projection = ''
    for i, bins in enumerate(n_bins):

        dim = 4 - bins.count(1)
        if i > 0:
            projection += '_and_'
        projection += str(dim) + 'd_'

        if bins.count(1) == 0 and i == 0:  # for 4D input # TODO FIX BUG XYZT AFTER NAME
            if swap_4d_channels is not None:
                projection += swap_4d_channels
            else:
                projection += 'xyz-c' if bins[3] == 31 else 'xyz-t'

        else:  # 2D/3D input
            if bins[0] > 1: projection += 'x'
            if bins[1] > 1: projection += 'y'
            if bins[2] > 1: projection += 'z'
            if bins[3] > 1: projection += 't'

    str_ident = '_' + str_ident if str_ident is not '' else str_ident
    modelname += projection + '_' + class_type[1] + str_ident

    return modelname


def orca_eval(cfg):
    """
    Core code that evaluates a neural network. The input parameters are the same as for orca_train, so that it is compatible
    with the .toml file.
    TODO Should be directly callable on a saved model, so that less arguments are required, and maybe no .toml is needed?

    """
    folder_name = cfg.main_folder
    test_files = cfg.get_val_files()
    n_bins = cfg.get_n_bins()
    class_type = cfg.class_type
    swap_4d_channels = cfg.swap_4d_channels
    batchsize = cfg.batchsize
    str_ident = cfg.str_ident
    list_name = os.path.basename(cfg.get_list_file()).split(".")[0]
    nn_arch = cfg.get_modeldata().nn_arch

    epoch = (cfg.initial_epoch, cfg.initial_fileno)
    if epoch[0] == -1 and epoch[1] == -1:
        epoch = cfg.get_latest_epoch()
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))

    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg)
    else:
        xs_mean = None

    if cfg.use_scratch_ssd:
        cfg.use_local_node()

    path_of_model = folder_name + 'saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
    model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident)
    arr_filename = folder_name + 'predictions/pred_model_epoch_{}_file_{}_on_{}_val_files.npy'.format(str(epoch[0]), str(epoch[1]), list_name)

    predict_and_investigate_model_performance(cfg, model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean, arr_filename, folder_name)


def example_run(main_folder, list_file, config_file, model_file):
    """
    This shows how to use OrcaNet.

    Parameters
    ----------
    main_folder : str
        Path to the folder where everything gets saved to, e.g. the summary log file, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and validation.
    config_file : str
        Path to a .toml file which overwrite some of the default settings for training and validating a model.
    model_file : str
        Path to a file with parameters to build a model of a predefined architecture with OrcaNet.

    """
    # Set up the cfg object with the input data
    cfg = Settings(main_folder, list_file, config_file)
    # Currently, the eval scripts are only supported for automatically generated models, so nn_arch is needed.
    cfg.set_from_model_file(model_file)
    orca_eval(cfg)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    main_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    model_file = args['MODEL']
    example_run(main_folder, list_file, config_file, model_file)


if __name__ == '__main__':
    parse_input()
