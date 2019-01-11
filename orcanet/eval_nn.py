#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Main code for evaluating NN's.
It can also be called via a parser by running this python module as follows:

Usage:
    run_nn.py CONFIG LIST [FOLDER]
    run_nn.py (-h | --help)

Arguments:
    CONFIG  A .toml file which sets up the model and training.
            An example can be found in config/models/example_model.toml
    LIST    A .list file which contains the files to be trained on.
            An example can be found in config/lists/example_list.toml
    FOLDER  A new subfolder will be generated in this folder, where everything from this model gets saved to.
            Default is the current working directory.

Options:
    -h --help                       Show this screen.

"""

import matplotlib as mpl
from docopt import docopt
mpl.use('Agg')

from utilities.input_output_utilities import use_node_local_ssd_for_input, read_out_list_file, read_out_config_file, look_for_latest_epoch, h5_get_n_bins
from utilities.nn_utilities import load_zero_center_data, get_modelname
from utilities.evaluation_utilities import *
from utilities.losses import get_all_loss_functions


def predict_and_investigate_model_performance(model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
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
        arr_nn_pred = get_nn_predictions_and_mc_info(model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, samples=None)
        print("Done! Saving prediction as:\n   "+arr_filename)
        np.save(arr_filename, arr_nn_pred)


    #arr_nn_pred = np.load('results/plots/saved_predictions/arr_nn_pred_' + modelname + '_final_stateful_false.npy')
    #arr_nn_pred = np.load('results/plots/saved_predictions//arr_nn_pred_model_VGG_4d_xyz-t_and_yzt-x_and_4d_xyzt_track-shower_multi_input_single_train_tight-1_tight-2_lr_0.003_tr_st_test_st_final_stateful_false_1-100GeV_precut.npy')

    if class_type[1] == 'track-shower':  # categorical
        precuts = (False, '3-100_GeV_prod')

        make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title='Classified as track', filename=folder_name + '/plots/ts_' + modelname,
                                                      precuts=precuts, corr_cut_pred_0=0.5)

        make_prob_hists(arr_nn_pred, folder_name, modelname=modelname, precuts=precuts)
        make_hist_2d_property_vs_property(arr_nn_pred, folder_name, modelname, property_types=('bjorken-y', 'probability'),
                                          e_cut=(1, 100), precuts=precuts)
        calculate_and_plot_separation_pid(arr_nn_pred, folder_name, modelname, precuts=precuts)

    else:  # regression
        arr_nn_pred_shallow = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')
        #precuts = (True, 'regr_3-100_GeV_prod_and_1-3_GeV_prod')
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

            make_1d_reco_err_div_by_std_plot(arr_nn_pred, modelname, folder_name, precuts=precuts) # TODO take precuts from above?
            make_1d_reco_err_to_reco_residual_plot(arr_nn_pred, modelname, folder_name, precuts=precuts)
            make_2d_dir_correlation_plot_different_sigmas(arr_nn_pred, modelname, folder_name, precuts=precuts)


def eval_nn(list_filename, folder_name, loss_opt, class_type, nn_arch,
               swap_4d_channels=None, batchsize=64, epoch=[-1,-1], epochs_to_train=-1, n_gpu=(1, 'avolkov'), use_scratch_ssd=False,
               zero_center=False, shuffle=(False,None), str_ident='', train_logger_display=100, train_logger_flush=-1,
               train_verbose=2, n_events=None):
    """
    Core code that evaluates a neural network. The input parameters are the same as for orca_train, so that it is compatible
    with the .toml file.
    TODO Should be directly callable on a saved model, so that less arguments are required, and maybe no .toml is needed?

    """

    train_files, test_files, multiple_inputs = read_out_list_file(list_filename)
    n_bins = h5_get_n_bins(train_files)
    if epoch == [-1, -1]:
        epoch = look_for_latest_epoch(folder_name)
        print("Automatically initialized epoch to epoch {} file {}.".format(epoch[0], epoch[1]))
    if zero_center:
        xs_mean = load_zero_center_data(train_files, batchsize, n_bins, n_gpu[0])
    else:
        xs_mean = None
    if use_scratch_ssd:
        train_files, test_files = use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=multiple_inputs)

    path_of_model = folder_name + '/saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
    model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident)
    arr_filename = folder_name + '/predictions/pred_model_epoch_{}_file_{}_on_{}.npy'.format(str(epoch[0]), str(epoch[1]), list_filename[:-5].split("/")[-1])

    predict_and_investigate_model_performance(model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean, arr_filename, folder_name)


def orca_eval(trained_models_folder, config_file, list_file):
    """
    Frontend function for evaluating networks.

    Parameters
    ----------
    trained_models_folder : str
        Path to the folder where everything gets saved to.
        Every model (from a .toml file) will get its own folder in here, with the name being the
        same as the one from the .toml file.
    config_file : str
        Path to a .toml file which contains all the infos for training and testing of a model.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training.

    """
    keyword_arguments = read_out_config_file(config_file)
    folder_name = trained_models_folder + str(os.path.splitext(os.path.basename(config_file))[0])
    eval_nn(list_file, folder_name, **keyword_arguments)


def parse_input():
    """ Run the orca_eval function with a parser. """
    args = docopt(__doc__)
    config_file = args['CONFIG']
    list_file = args['LIST']
    trained_models_folder = args['FOLDER'] if args['FOLDER'] is not None else "./"
    orca_eval(trained_models_folder, config_file, list_file)


if __name__ == '__main__':
    parse_input()