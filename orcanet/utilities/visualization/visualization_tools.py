# -*- coding: utf-8 -*-
"""
Visualization tools used with Keras.
1) Makes performance graphs for training and testing.
2) Visualizes activations for Keras models
"""
import inspect
import numpy as np
import keras as ks
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utilities.nn_utilities import generate_batches_from_hdf5_file
from utilities.losses import get_all_loss_functions


def plot_train_and_test_statistics_old(modelname, model, folder_name):
    """
    Plots the loss in training/testing based on .txt logfiles.
    :param str modelname: name of the model.
    :param ks.model.Model model: Keras model of the neural network.
    :param str folder_name: Path of the main folder of this model.
    """
    # #Batch number # BatchNumber float # Losses # Metrics
    log_array_train = np.loadtxt(folder_name + '/log_train_' + modelname + '.txt', dtype=np.float32, delimiter='\t', skiprows=1, ndmin=2)
    # #Epoch # Losses # Metrics
    log_array_test = np.loadtxt('models/trained/perf_plots/log_test_' + modelname + '.txt', dtype=np.float32, delimiter='\t', skiprows=1, ndmin=2)

    fig, axes = plt.subplots()
    colors = ['#000000', '#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77',
              '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466', '#4477AA'] # ref. personal.sron.nl/~pault/
    pdf_plots = PdfPages('models/trained/perf_plots/plots/loss_' + modelname + '.pdf')

    skip_n_first_batches, batchlogger_display = 500, 100
    first_line = int(skip_n_first_batches / batchlogger_display)

    train_batchnr = log_array_train[first_line:, 1]
    test_epoch = log_array_test[:, 0]

    x_ticks_major = get_epoch_xticks(test_epoch, train_batchnr)

    i, j = 0, 0
    for metric_name in model.metrics_names: # metric names have same order as the columns in the log_array_train/test
        if 'loss' in metric_name:

            i += 1
            train_metric_loss = log_array_train[first_line:, 1 + i] # skip a certain number of train batches for better y scale
            test_metric_loss = log_array_test[:, 0 + i]

            if 'err' in metric_name:
                j += 1
                color = colors[j]
            else:
                color = colors[i - 1]

            plt.plot(train_batchnr, train_metric_loss, color=color, ls='--', zorder=3, label='train, ' + metric_name, lw=0.5, alpha=0.5)
            plt.plot(test_epoch, test_metric_loss, color=color, marker='o', zorder=3, label='val, ' + metric_name, lw=0.5, markersize=3)

            plt.xticks(x_ticks_major)
            test_metric_min_to_max = np.amax(test_metric_loss) - np.amin(test_metric_loss)
            y_lim = (np.amin(test_metric_loss) - 0.25 * test_metric_min_to_max, np.amax(test_metric_loss) + 0.25 * test_metric_min_to_max)
            plt.ylim(y_lim)

            axes.legend(loc='upper right')
            plt.xlabel('Epoch [#]')
            plt.ylabel('Loss')
            title = plt.title('Loss for ' + modelname)
            title.set_position([.5, 1.04])
            plt.grid(True, zorder=0, linestyle='dotted')

            pdf_plots.savefig(fig)
            plt.savefig('models/trained/perf_plots/plots/png/loss_' + metric_name + '_' + modelname + '.png', dpi=600)
            plt.cla()

    plt.close()
    pdf_plots.close()


def make_test_train_plot(test_train_data_list, title=""):
    """
    Plot one or more test/train lines in a single plot.

    Parameters
    ----------
    test_train_data_list : list
        test_train_data_list[0] = x and y test data as a list, plotted as connected dots
        test_train_data_list[1] = x and y train data as a list, plotted as a faint line
        test_train_data_list[2] = label used for the train/test lines
        test_train_data_list[3] = color used for the train/test lines
    title : str
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.
    """
    fig, ax = plt.subplots()
    all_x_coordinates_test, all_x_coordinates_train = [],[]
    all_y_coordinates_test, all_y_coordinates_train = [],[]
    for test_train_data in test_train_data_list:
        test_data, train_data, label, color = test_train_data
        plt.plot(test_data[0], test_data[1], color=color, marker='o', zorder=3, label='test ' + label)
        plt.plot(train_data[0], train_data[1], color=color, ls='--', zorder=3, label='train ' + label,
                 lw=0.6, alpha=0.5)
        all_x_coordinates_test.extend(test_data[0])
        all_x_coordinates_train.extend(train_data[0])
        all_y_coordinates_test.extend(test_data[1])
        all_y_coordinates_train.extend(train_data[1])

    plt.xticks(get_epoch_xticks(all_x_coordinates_test+all_x_coordinates_train))
    test_metric_min_to_max = np.amax(all_y_coordinates_test) - np.amin(all_y_coordinates_test)
    y_lim = (np.amin(all_y_coordinates_test) - 0.25 * test_metric_min_to_max,
             np.amax(all_y_coordinates_test) + 0.25 * test_metric_min_to_max)
    plt.ylim(y_lim)

    ax.legend(loc='upper right')
    plt.xlabel('Epoch [#]')
    plt.ylabel('Loss')
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0, linestyle='dotted')

    return fig

def plot_metrics(summary_data, full_train_data, metric_names="loss", make_auto_titles=False):
    """
    Plot one or more metrics over the epochs from a summary.txt file in a single plot,
    each with its test and train curves.

    Parameters
    ----------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files, merged into a single array.
    metric_names : list or str
        Name or list of names of metrics to be plotted over the epoch. The name is what was written in the head line
        of the summary file, except without the train_ or test_ prefix.
    make_auto_titles : bool
        If true, the title of the plot will be the name of the first metric in metric_names.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.
    """
    test_train_data_list = []
    if isinstance(metric_names, str): metric_names=[metric_names,]
    for metric_name in metric_names:
        summary_label   = "test_"+metric_name
        train_log_label = metric_name

        test_data = [summary_data["Epoch"], summary_data[summary_label]]
        train_data = [full_train_data["Batch_float"], full_train_data[train_log_label]]
        label = metric_name
        color = "red"
        test_train_data = test_data, train_data, label, color
        test_train_data_list.append(test_train_data)
    if make_auto_titles:
        title = metric_name
    else:
        title = ""
    fig = make_test_train_plot(test_train_data_list, title)
    return fig

def get_epoch_xticks(x_coordinates):
    """
    Calculates the xticks for the train and test statistics matplotlib plot.

    Parameters
    ----------
    x_coordinates : list
        List of the x-coordinates of the points in the plot.

    Returns
    -------
    x_ticks_major : numpy.ndarray
        Array containing the ticks.

    """
    minimum, maximum = np.amin(x_coordinates), np.amax(x_coordinates)
    start_epoch, end_epoch = np.floor(minimum), np.ceil(maximum)
    # reduce number of x_ticks by factor of 2 if n_epochs > 20
    n_epochs = end_epoch - start_epoch
    x_ticks_stepsize = 1 + np.floor(n_epochs / 20.) # 20 ticks max, increase stepsize if n_epochs >= 20
    x_ticks_major = np.arange(start_epoch, end_epoch + x_ticks_stepsize, x_ticks_stepsize)

    return x_ticks_major

def plot_all_metrics_to_pdf(summary_data, full_train_data, pdf_name):
    """
    Plot all metrics of the given data into a pdf file, each metric in its own plot.

    Parameters
    ----------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files, merged into a single array.
    pdf_name : str
        Where the pdf will get saved.

    """
    all_metrics = []
    for keyword in summary_data.dtype.names:
        if keyword == "Epoch" or keyword=="LR":
            continue
        if "train_" in keyword:
            keyword = keyword.split("train_")[-1]
        else:
            keyword = keyword.split("test_")[-1]
        if not keyword in all_metrics:
            all_metrics.append(keyword)
    with PdfPages(pdf_name) as pdf:
        for metric in all_metrics:
            fig = plot_metrics(summary_data, full_train_data, metric_names=metric, make_auto_titles=True)
            pdf.savefig(fig)
            plt.close(fig)

#plot_all_metrics_to_pdf(a[0],a[1],"test.pdf")

def get_activations_and_weights(f, n_bins, class_type, xs_mean, swap_4d_channels, modelname, epoch, str_ident, file_no=1, layer_name=None, learning_phase='test'):
    """
    Get the weights of a model and also the activations of the model for a single event.
    :param str f: path to a .h5 file that contains images of events. Needed for plotting the activations for the event.
    :param list(tuple) n_bins: the number of bins for each dimension (x,y,z,t) in the supplied file. Can contain multiple n_bins tuples.
    :param (int, str) class_type: the number of output classes and a string identifier to specify the exact output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the data.
    :param None/int swap_4d_channels: for 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str modelname: Name of the model in order to load it.
    :param int epoch: Epoch of the trained model.
    :param str str_ident: string identifier that is parsed to the generator. Needed for some projection types.
    :param int file_no: File Number of the trained model in this epoch (if multiple files are trained per epoch).
    :param None/str layer_name: if only the activations of a single layer should be collected.
    :param str learning_phase: string identifier to specify the learning phase during the calculation of the activations.
                               'test', 'train': Dropout, Batchnorm etc. in test/train mode
    """
    lp = 0. if learning_phase == 'test' else 1.

    generator = generate_batches_from_hdf5_file(f, 1, n_bins, class_type, str_ident, zero_center_image=xs_mean, swap_col=swap_4d_channels, yield_mc_info=True)
    model_inputs, ys, y_values = next(generator) # y_values = mc_info for the event

    custom_objects = get_all_loss_functions()
    saved_model = ks.models.load_model('models/trained/trained_' + modelname + '_epoch_' + str(epoch) + '_file_' + str(file_no) + '.h5',
                                       custom_objects=custom_objects)

    inp = saved_model.input
    model_multi_inputs_cond = True if len(model_inputs) > 1 else False

    if not isinstance(inp, list):
        inp = [inp] # only one input! let's wrap it in a list.

    outputs = [layer.output for layer in saved_model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs -> empty tf.tensors
    layer_names = [layer.name for layer in saved_model.layers if
                   layer.name == layer_name or layer_name is None]
    weights = [layer.get_weights() for layer in saved_model.layers if
               layer.name == layer_name or layer_name is None]

    outputs = outputs[1:] # remove the first input_layer from fetch
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(lp)
    else:
        list_inputs = [model_inputs, lp]

    layer_outputs = [func(list_inputs)[0] for func in funcs]
    activations = []
    for layer_activations in layer_outputs:
        activations.append(layer_activations)

    return layer_names, activations, weights, y_values


def plot_weights_and_activations(f, n_bins, class_type, xs_mean, swap_4d_channels, modelname, epoch, file_no, str_ident):
    """
    Plots the weights of a model and the activations for one event to a .pdf file.
    :param str f: path to a .h5 file that contains images of events. Needed for plotting the activations for the event.
    :param list(tuple) n_bins: the number of bins for each dimension (x,y,z,t) in the supplied file. Can contain multiple n_bins tuples.
    :param (int, str) class_type: the number of output classes and a string identifier to specify the exact output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the data.
    :param None/int swap_4d_channels: for 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str modelname: name of the model.
    :param int epoch: epoch of the model.
    :param int file_no: File Number of the trained model in this epoch (if multiple files are trained per epoch).
    :param str str_ident: string identifier that is parsed to the get_activations_and_weights function. Needed for some projection types.
    """
    layer_names, activations, weights, y_values = get_activations_and_weights(f, n_bins, class_type, xs_mean, swap_4d_channels,
                                                                              modelname, epoch, str_ident, file_no=file_no,  layer_name=None, learning_phase='test')

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
    plt.close()