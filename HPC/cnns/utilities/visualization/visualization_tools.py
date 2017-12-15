# -*- coding: utf-8 -*-
"""Visualization tools for activations in Keras.
"""

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..cnn_utilities import generate_batches_from_hdf5_file


def plot_train_and_test_statistics(modelname):
    """
    Plots the loss in training/testing based on .txt logfiles.
    :param str modelname: name of the model.
    """
    # #Batch number # BatchNumber float # Loss # Accuracy
    log_array_train = np.loadtxt('models/trained/perf_plots/log_train_' + modelname + '.txt', dtype=np.float32, delimiter='\t', skiprows=1, ndmin=2)
    # #Epoch # Loss # Accuracy
    log_array_test = np.loadtxt('models/trained/perf_plots/log_test_' + modelname + '.txt', dtype=np.float32, delimiter='\t', skiprows=1, ndmin=2)

    train_batchnr = log_array_train[:, 1]
    train_loss = log_array_train[:, 2]
    test_epoch = log_array_test[:, 0]
    test_loss = log_array_test[:, 1]

    fig, axes = plt.subplots()

    # plot loss statistics
    plt.plot(train_batchnr, train_loss, 'b--', zorder=3, label='train', lw=0.5, alpha=0.5)
    plt.plot(test_epoch, test_loss, 'b', marker='o', zorder=3, label='test', lw=0.5, markersize=3)

    x_ticks_major = get_epoch_xticks(test_epoch, train_batchnr)
    plt.xticks(x_ticks_major)

    axes.legend(loc='upper right')
    plt.xlabel('Epoch [#]')
    plt.ylabel('Loss')
    title = plt.title('Loss for ' + modelname)
    title.set_position([.5, 1.04])

    plt.grid(True, zorder=0, linestyle='dotted')

    plt.savefig('models/trained/perf_plots/plots/loss_' + modelname + '.pdf')
    plt.savefig('models/trained/perf_plots/plots/loss_' + modelname + '.png', dpi=600)


def get_epoch_xticks(test_epoch, train_batchnr):

    # if we didn't start logging with epoch 1
    min_test, max_test = np.amin(test_epoch), np.amax(test_epoch)
    min_train, max_train = np.amin(train_batchnr), np.amax(train_batchnr)
    minimum = min_test if min_test < min_train else min_train
    maximum = max_test if max_test > max_train else max_train
    start_epoch, end_epoch = np.floor(minimum), np.ceil(maximum)

    # reduce number of x_ticks by factor of 2 if n_epochs > 20
    n_epochs = end_epoch - start_epoch
    #reduce_x_ticks = 1 + np.floor(n_epochs / 10.)
    #x_ticks_n_steps = (n_epochs + 2) / reduce_x_ticks

    #x_ticks_major = np.linspace(start_epoch, end_epoch + 1, x_ticks_n_steps)
    x_ticks_stepsize = 1 + np.floor(n_epochs / 20.) # 20 ticks max, increase stepsize if n_epochs >= 20
    x_ticks_major = np.arange(start_epoch, end_epoch + x_ticks_stepsize, x_ticks_stepsize)

    return x_ticks_major



def get_activations_and_weights(model, f, n_bins, class_type, xs_mean, swap_4d_channels,  layer_name=None, learning_phase='test'):
    """
    Get the weights of a model and also the activations of the model for a single event.
    :param ks.model.Model model: trained Keras model of a neural network.
    :param str f: path to a .h5 file that contains images of events. Needed for plotting the activations for the event.
    :param tuple n_bins: the number of bins for each dimension (x,y,z,t) in the supplied file.
    :param (int, str) class_type: the number of output classes and a string identifier to specify the exact output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the data.
    :param None/int swap_4d_channels: for 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param None/str layer_name: if only the activations of a single layer should be collected.
    :param str learning_phase: string identifier to specify the learning phase during the calculation of the activations.
                               'test', 'train': Dropout, Batchnorm etc. in test/train mode
    """
    lp = 0. if learning_phase == 'test' else 1.

    generator = generate_batches_from_hdf5_file(f, 1, n_bins, class_type, zero_center_image=xs_mean, swap_col=swap_4d_channels, yield_mc_info=True)
    model_inputs, ys, y_values = next(generator) # y_values = mc_info for the event

    inp = model.input

    model_multi_inputs_cond = True if isinstance(inp, list) else False
    if not isinstance(inp, list):
        inp = [inp] # only one input! let's wrap it in a list.

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs -> empty tf.tensors
    layer_names = [layer.name for layer in model.layers if
                   layer.name == layer_name or layer_name is None]
    weights = [layer.get_weights() for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(lp)
    else:
        list_inputs = [model_inputs, lp]

    layer_outputs = [func(list_inputs)[0] for func in funcs]
    activations = []
    for layer_activations in layer_outputs: #TODO layer_outputs == activations??
        activations.append(layer_activations)

    return layer_names, activations, weights, y_values


def plot_weights_and_activations(model, f, n_bins, class_type, xs_mean, swap_4d_channels, modelname, epoch):
    """
    Plots the weights of a model and the activations for one event to a .pdf file.
    :param ks.model.Model model: trained Keras model of a neural network.
    :param str f: path to a .h5 file that contains images of events. Needed for plotting the activations for the event.
    :param tuple n_bins: the number of bins for each dimension (x,y,z,t) in the supplied file.
    :param (int, str) class_type: the number of output classes and a string identifier to specify the exact output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the data.
    :param None/int swap_4d_channels: for 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str modelname: name of the model.
    :param int epoch: epoch of the model.
    """
    layer_names, activations, weights, y_values = get_activations_and_weights(model, f, n_bins, class_type, xs_mean, swap_4d_channels,  layer_name=None, learning_phase='test')

    fig, axes = plt.subplots()
    pdf_activations_and_weights = PdfPages('models/trained/perf_plots/model_stat_plots/act_and_weights_plots_' + modelname + '_epoch' + str(epoch) + '.pdf')

    # plot weights
    event_id = int(y_values[0][0])
    energy = y_values[0][2]

    try:
        run_id = int(y_values[0][9]) # if it doesn't exist in the file
    except IndexError:
        run_id = ''

    for i, layer_activations in enumerate(activations):
        plt.hist(layer_activations.flatten(), bins=100)
        plt.title('Run/Event-ID: ' + str(run_id) + '/' + str(event_id) + ', E=' + str(energy) + 'GeV. \n ' + 'Activations for layer ' + str(layer_names[i]))
        plt.xlabel('Activation (layer output)')
        plt.ylabel('Quantity [#]')
        pdf_activations_and_weights.savefig(fig)
        plt.cla()

    for i, layer_weights in enumerate(weights):
        w = None

        if not layer_weights: continue  # skip if layer weights are empty
        for j, w_temp in enumerate(layer_weights):
            # ignore different origins of the weights
            if j == 0:
                w = np.array(w_temp.flatten(), dtype=np.float64)
            else:
                w_temp_flattened = np.array(w_temp.flatten(), dtype=np.float64)
                w = np.concatenate((w, w_temp_flattened), axis=0)

        plt.hist(w, bins=100)
        plt.title('Weights for layer ' + str(layer_names[i]))
        plt.xlabel('Weight')
        plt.xlabel('Quantity [#]')
        plt.tight_layout()
        pdf_activations_and_weights.savefig(fig)
        plt.cla()

    pdf_activations_and_weights.close()


# def test_plotting():
#     # for testing
#     import keras as ks
#
#     batchsize = 32 # doesn't matter here
#     n_gpu = 1  # doesn't matter here
#
#     train_file = '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/with_run_id/without_mc_time_fix/h5/xyzt/concatenated/elec-CC_and_muon-CC_xyzt_train_1_to_480_shuffled_0.h5'
#     test_file = '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/with_run_id/without_mc_time_fix/h5/xyzt/concatenated/elec-CC_and_muon-CC_xyzt_test_481_to_600_shuffled_0.h5'
#
#     epoch = 1
#     n_bins = (11, 13, 18, 60)
#     class_type = (2, 'muon-CC_to_elec-CC')
#     swap_4d_channels = 'yzt-x'
#     nn_arch = 'VGG'
#     from ..cnn_utilities import get_modelname
#     modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels)
#     from ..cnn_utilities import load_zero_center_data
#     xs_mean = load_zero_center_data([[train_file]], batchsize, n_bins, n_gpu, swap_4d_channels=swap_4d_channels)
#
#     model = ks.models.load_model('models/trained/trained_' + modelname + '_epoch' + str(epoch) + '.h5')
#
#     plot_weights_and_activations(model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean,
#                                  swap_4d_channels='yzt-x', modelname=modelname, epoch=epoch)
#
#
# if __name__ == '__main__':
#     from os import sys, path
#     from cnn_utilities import generate_batches_from_hdf5_file
#     sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
#     test_plotting()
