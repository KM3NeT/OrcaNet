# -*- coding: utf-8 -*-
"""
Visualization tools used with Keras.
1) Makes performance graphs for training and validating.
2) Visualizes activations for Keras models
"""
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from orcanet.utilities.nn_utilities import generate_batches_from_hdf5_file, get_inputs


def make_test_train_plot(train_data, val_data=None, label=None, color=None, title=""):
    """
    Plot a val/train line in a single plot.

    The valdata can contain nan's, or might even be None.

    Parameters
    ----------
    train_data : list
        X data (0) and y data (1) of the train curve. Will be plotted as connected dots.
    val_data : list or None
        X data (0) and y data (1) of the validation curve. Will be plotted as a faint solid line.
    label : str
        Labels used for the train/val line.
    color : str or None
        Colors used for the train/val line.
    title : str
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.

    """
    fig, ax = plt.subplots()

    train_plot = plt.plot(train_data[0], train_data[1], color=color, ls='--', zorder=3, label='train ' + label, lw=0.6, alpha=0.5)
    if val_data is not None:
        # val plot always has the same color as the train plot
        plt.plot(val_data[0], val_data[1], color=train_plot[0].get_color(), marker='o', zorder=3, label='val ' + label)

    all_x_coordinates_test.extend(val_data[0])
    all_x_coordinates_train.extend(train_data[0])
    # Remove the occasional np.nan from the y data
    all_y_coordinates_test.extend(val_data[1][~np.isnan(val_data[1])])
    all_y_coordinates_train.extend(train_data[1][~np.isnan(train_data[1])])

    # TODO also incorporate the train curve somehow into the limits
    if not len(all_y_coordinates_test) == 0:
        test_metric_min_to_max = np.amax(all_y_coordinates_test) - np.amin(all_y_coordinates_test)
        y_lim = (np.amin(all_y_coordinates_test) - 0.25 * test_metric_min_to_max,
                 np.amax(all_y_coordinates_test) + 0.25 * test_metric_min_to_max)
    else:
        test_metric_min_to_max = np.amax(all_y_coordinates_train) - np.amin(all_y_coordinates_train)
        y_lim = (np.amin(all_y_coordinates_train) - 0.25 * test_metric_min_to_max,
                 np.amax(all_y_coordinates_train) + 0.25 * test_metric_min_to_max)

    plt.ylim(y_lim)
    plt.xticks(get_epoch_xticks(all_x_coordinates_test + all_x_coordinates_train))

    ax.legend(loc='upper right')
    plt.xlabel('Epoch [#]')
    plt.ylabel('Loss')
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0, linestyle='dotted')

    return fig


def plot_metric(summary_data, full_train_data, metric_name="loss", make_auto_titles=False, color=None):
    """
    Plot and return the training and validation history of a metric over the epochs in a single plot.

    Parameters
    ----------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files from ./log_train/, merged into a single array.
    metric_name : str
        Name of the metric to be plotted over the epoch. This name is what was written in the head line
        of the summary.txt file, except without the train_ or val_ prefix (as these will be added by this script).
    make_auto_titles : bool
        If true, the title of the plot will be the name of the metric.
    color : None or str
        The color of the train and val lines. If None is given, the default color cycle is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.

    """
    # Gather the Epoch and y data from the files
    label = metric_name
    summary_label = "val_"+metric_name
    if summary_data["Epoch"].shape == (0,):
        # When no lines are present in the summary.txt file.
        val_data = None
    elif summary_data["Epoch"].shape == ():
        # When only one line is present in the summary.txt file.
        val_data = [summary_data["Epoch"].reshape(1), summary_data[summary_label].reshape(1)]
    else:
        val_data = [summary_data["Epoch"], summary_data[summary_label]]
    train_data = [full_train_data["Batch_float"], full_train_data[metric_name]]

    if make_auto_titles:
        title = metric_name
    else:
        title = ""
    fig = make_test_train_plot(train_data, val_data, label, color, title)
    return fig


def get_epoch_xticks(x_coordinates):
    """
    Calculates the xticks for the train and validation statistics matplotlib plot.

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
    x_ticks_stepsize = 1 + np.floor(n_epochs / 20.)  # 20 ticks max, increase stepsize if n_epochs >= 20
    x_ticks_major = np.arange(start_epoch, end_epoch + x_ticks_stepsize, x_ticks_stepsize)

    return x_ticks_major


def plot_all_metrics_to_pdf(summary_data, full_train_data, pdf_name):
    """
    Plot and save all metrics of the given validation- and train-data from the training of a model
    into a pdf file, each metric in its own plot. If metric pairs of a variable and its error are found (e.g. e_loss
    and e_err_loss), they will have the same color and appear back to back in the plot.

    Parameters
    ----------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files, merged into a single array.
    pdf_name : str
        Where the pdf will get saved.

    """
    plt.ioff()
    # Extract the names of the metrics
    all_metrics = []
    for keyword in summary_data.dtype.names:
        if keyword == "Epoch" or keyword == "LR":
            continue
        if "train_" in keyword:
            keyword = keyword.split("train_")[-1]
        else:
            keyword = keyword.split("val_")[-1]
        if keyword not in all_metrics:
            all_metrics.append(keyword)
    all_metrics = sort_metric_names_and_errors(all_metrics)
    # Plot them
    colors = ['#000000', '#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77',
              '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
              '#4477AA']  # ref. personal.sron.nl/~pault/
    color_counter = 0
    with PdfPages(pdf_name) as pdf:
        for metric_no, metric in enumerate(all_metrics):
            # If this metric is an err metric of a variable, color it the same
            if all_metrics[metric_no-1] == metric.replace("_err", ""):
                color_counter -= 1
            fig = plot_metric(summary_data, full_train_data, metric_names=metric, make_auto_titles=True, color=colors[color_counter % len(colors)])
            color_counter += 1
            pdf.savefig(fig)
            plt.close(fig)


def sort_metric_names_and_errors(metric_names):
    """
    Sort a list of metrices, so that errors are right after their variable.
    The format of the metric names have to be e.g. e_loss and e_err_loss for this to work.

    Example
    ----------
    >>> sort_metric_names_and_errors( ['e_loss', 'loss', 'e_err_loss', 'dx_err_loss'] )
    ['e_loss', 'e_err_loss', 'loss', 'dx_err_loss']

    Parameters
    ----------
    metric_names : List
        List of metric names.

    Returns
    -------
    sorted_metrics : List
        List of sorted metric names with the same length as the input.

    """
    sorted_metrics = [0] * len(metric_names)
    counter = 0
    for metric_name in metric_names:
        if "err_" in metric_name:
            if metric_name.replace("err_", "") not in metric_names:
                sorted_metrics[counter] = metric_name
                counter += 1
            continue
        sorted_metrics[counter] = metric_name
        counter += 1
        err_loss = metric_name.split("_loss")[0]+"_err_loss"
        if err_loss in metric_names:
            sorted_metrics[counter] = err_loss
            counter += 1
    if 0 in sorted_metrics:
        print("Warning: Something went wrong with the sorting of metrics! Given was {}, output was {}. Using unsorted metrics instead.".format(metric_names, sorted_metrics))
        sorted_metrics = metric_names
    return sorted_metrics


def get_activations_and_weights(cfg, xs_mean, model, layer_name=None, learning_phase='test'):
    """
    Get the weights of a model and also the activations of the model for a single event in the validation set.

    xs_mean : dict
        Mean image of the dataset used for zero-centering. Every input as a key, ndarray as values.
    :param ks.model model: The model to make the data with.
    :param None/str layer_name: if only the activations of a single layer should be collected.
    :param str learning_phase: string identifier to specify the learning phase during the calculation of the activations.
                               'test', 'train': Dropout, Batchnorm etc. in test/train mode
    """
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs -> empty tf.tensors
    layer_names = [layer.name for layer in model.layers if
                   layer.name == layer_name or layer_name is None]
    weights = [layer.get_weights() for layer in model.layers if
               layer.name == layer_name or layer_name is None]
    outputs = outputs[1:]  # remove the first input_layer from fetch

    # get input layers and input files
    inputs = get_inputs(model)
    f = next(cfg.yield_files("val"))
    generator = generate_batches_from_hdf5_file(cfg, f, f_size=1, zero_center_image=xs_mean, yield_mc_info=True)
    model_inputs, ys, y_values = next(generator)  # y_values = mc_info for the event

    # doesnt work with dicts for whatever reason so transform into lists instead
    keys = list(inputs.keys())
    inp = [inputs[key] for key in keys]
    model_inputs = [model_inputs[key] for key in keys]
    if len(inp) == 1:
        inp = inp[0]
        model_inputs = model_inputs[0]

    funcs = [K.function([inp, K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    lp = 0. if learning_phase == 'test' else 1.
    list_inputs = [model_inputs, lp]

    layer_outputs = [func(list_inputs)[0] for func in funcs]
    activations = []
    for layer_activations in layer_outputs:
        activations.append(layer_activations)

    return layer_names, activations, weights, y_values


def plot_weights_and_activations(cfg, model, xs_mean, epoch):
    """
    Plots the weights of a model and the activations for one event to a .pdf file.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.models.Model
        The model to do the predictions with.
    xs_mean : dict
        Mean image of the dataset used for zero-centering. Every input as a key, ndarray as values.
    epoch : tuple
        Current epoch and fileno.

    """
    layer_names, activations, weights, y_values = get_activations_and_weights(cfg, xs_mean, model, layer_name=None, learning_phase='test')

    fig, axes = plt.subplots()
    pdf_name = cfg.get_subfolder("activations", create=True) + "/act_and_weights_plots_epoch_" + str(epoch) + '.pdf'
    pdf_activations_and_weights = PdfPages(pdf_name)

    # plot weights
    event_id = int(y_values[0][0])
    energy = y_values[0][2]

    try:
        run_id = int(y_values[0][9])  # if it doesn't exist in the file
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

        if not layer_weights:
            continue  # skip if layer weights are empty
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
