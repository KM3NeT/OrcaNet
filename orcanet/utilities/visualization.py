# -*- coding: utf-8 -*-
"""
Visualization tools used with Keras.
1) Makes performance graphs for training and validating.
2) Visualizes activations for Keras models
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from orcanet.utilities.nn_utilities import get_layer_output


def make_train_val_plot(train_data, val_data=None, color=None, title=None,
                        y_label=None):
    """
    Plot a training and optionally a validation line in a single plot.

    The val data can contain nan's.

    Parameters
    ----------
    train_data : List
        X data [0] and y data [1] of the train curve. Will be plotted as
        connected dots.
    val_data : List or None
        X data [0] and y data [1] of the validation curve. Will be plotted
        as a faint solid line.
    color : str or None
        Colors used for the train/val line.
    title : str or None
        Title of the plot.
    y_label : str or None
        Y label of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.

    """
    plt.ioff()
    fig, ax = plt.subplots()
    train_label, val_label = 'training', 'validation'

    train_plot = plt.plot(train_data[0], train_data[1], color=color, ls='--',
                          zorder=3, label=train_label, lw=0.6, alpha=0.5)
    if val_data is not None:
        # Skip over nan values, so that all dots are connected
        not_nan = ~np.isnan(val_data[1])
        val_data[0] = val_data[0][not_nan]
        val_data[1] = val_data[1][not_nan]
        # val plot always has the same color as the train plot
        plt.plot(val_data[0], val_data[1], color=train_plot[0].get_color(),
                 marker='o', zorder=3, label=val_label)

    plt.ylim(get_ylims(train_data, val_data, fraction=0.25))
    plt.xticks(get_epoch_xticks(train_data, val_data))

    ax.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    if title is not None:
        title = plt.title(title)
        title.set_position([.5, 1.04])
    plt.grid(True, zorder=0, linestyle='dotted')

    return fig


def get_ylims(train_data, val_data=None, fraction=0.25):
    """
    Get the y limits for the summary plot.

    Parameters
    ----------
    train_data : List
        X data (0) and y data (1) of the train curve. Will be plotted as
        connected dots.
    val_data : List or None
        X data (0) and y data (1) of the validation curve. Will be plotted
        as a faint solid line.
    fraction : float
        How much whitespace of the total y range is added above and below
        the lines.

    Returns
    -------
    y_lims : tuple
        Minimum, maximum of the data.

    """
    y_train = train_data[1]
    y_lims = np.amin(y_train), np.amax(y_train)

    if val_data is not None:
        y_val = val_data[1]
        y_lim_val = np.amin(y_val), np.amax(y_val)
        y_lims = (np.amin([y_lim_val[0], y_lims[0]]),
                  np.amax([y_lim_val[1], y_lims[1]]))

    y_range = y_lims[1] - y_lims[0]
    y_lims = (y_lims[0] - fraction * y_range,  y_lims[1] + fraction * y_range)
    return y_lims


def get_epoch_xticks(train_data, val_data=None):
    """
    Calculates the xticks for the train and validation summary plot.

    Parameters
    ----------
    train_data : List
        X data (0) and y data (1) of the train curve. Will be plotted as
        connected dots.
    val_data : List or None
        X data (0) and y data (1) of the validation curve. Will be plotted
        as a faint solid line.

    Returns
    -------
    x_ticks_major : numpy.ndarray
        Array containing the ticks.

    """
    x_values = train_data[0]
    if val_data is not None:
        x_values = np.append(x_values, val_data[0])

    minimum, maximum = np.amin(x_values), np.amax(x_values)
    start_epoch, end_epoch = np.floor(minimum), np.ceil(maximum)
    # reduce number of x_ticks by factor of 2 if n_epochs > 20
    n_epochs = end_epoch - start_epoch
    x_ticks_stepsize = 1 + np.floor(n_epochs / 20.)
    x_ticks_major = np.arange(start_epoch,
                              end_epoch + x_ticks_stepsize, x_ticks_stepsize)

    return x_ticks_major


def plot_metric(summary_data, full_train_data, metric_name="loss",
                title=None, color=None, y_label="loss"):
    """
    Plot and return the training and validation history of a metric over
    the epochs in a single plot.

    Parameters
    ----------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files
        from ./train_log/, merged into a single array.
    metric_name : str
        Name of the metric to be plotted over the epoch. This name is what
        was written in the head line of the summary.txt file, except without
        the train_ or val_ prefix.
    title : str or None
        Title of the plot.
    color : None or str
        The color of the train and val lines. If None is given, the default
        color cycle is used.
    y_label : str or None
        Y label of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.

    """
    summary_label = "val_"+metric_name

    assert metric_name in full_train_data.dtype.names, \
        "Train log metric name {} unknown, must be one of {}".format(
            metric_name, full_train_data.dtype.names)
    assert summary_label in summary_data.dtype.names, \
        "Summary metric name {} unknown, must be one of {}".format(
            summary_label, summary_data.dtype.names)

    if summary_data["Epoch"].shape == (0,):
        # When no lines are present in the summary.txt file.
        val_data = None
    elif summary_data["Epoch"].shape == ():
        # When only one line is present in the summary.txt file.
        val_data = [summary_data["Epoch"].reshape(1),
                    summary_data[summary_label].reshape(1)]
    else:
        val_data = [summary_data["Epoch"], summary_data[summary_label]]

    if np.all(np.isnan(val_data)[1]):
        val_data = None

    if full_train_data["Batch_float"].shape == (0,):
        # When no lines are present
        raise ValueError("Can not make summary plot: Training log files "
                         "contain no data!")
    elif full_train_data["Batch_float"].shape == ():
        # When only one line is present
        train_data = [full_train_data["Batch_float"].reshape(1),
                      full_train_data[metric_name].reshape(1)]
    else:
        train_data = [full_train_data["Batch_float"],
                      full_train_data[metric_name]]

    fig = make_train_val_plot(train_data, val_data, color=color, title=title,
                              y_label=y_label)
    return fig


def plot_all_metrics_to_pdf(summary_data, full_train_data, pdf_name):
    """
    Plot and save all metrics of the given validation- and train-data
    into a pdf file, each metric in its own plot.

    If metric pairs of a variable and its error are found (e.g. e_loss
    and e_err_loss), they will have the same color and appear back to
    back in the plot.

    Parameters
    ----------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files,
        merged into a single array.
    pdf_name : str
        Where the pdf will get saved.

    """
    plt.ioff()
    # Extract the names of the metrics
    all_metrics = get_all_metrics(summary_data)
    # Sort them
    all_metrics = sort_metrics(all_metrics)
    # Plot them w/ custom color cycle
    colors = ['#000000', '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
              '#DDCC77', '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC',
              '#AA4466', '#4477AA']  # ref. personal.sron.nl/~pault/
    color_counter = 0
    with PdfPages(pdf_name) as pdf:
        for metric_no, metric in enumerate(all_metrics):
            # If this metric is an err metric of a variable, color it the same
            if all_metrics[metric_no-1] == metric.replace("_err", ""):
                color_counter -= 1
            fig = plot_metric(summary_data, full_train_data,
                              metric_name=metric, title=metric,
                              color=colors[color_counter % len(colors)])
            color_counter += 1
            pdf.savefig(fig)
            plt.close(fig)


def get_all_metrics(summary_data):
    """ Get the name of the metrics from the first file in the summary.txt
    (not Epoch and LR. Also strip the train_ and val_) """
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
    return all_metrics


def sort_metrics(metric_names):
    """
    Sort a list of metrics, so that errors are right after their variable.
    The format of the metric names have to be e.g. e_loss and e_err_loss
    for this to work.

    Example
    ----------
    >>> sort_metrics( ['e_loss', 'loss', 'e_err_loss', 'dx_err_loss'] )
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
        print("Warning: Something went wrong with the sorting of metrics!"
              "Given was {}, output was {}. Using unsorted metrics "
              "instead.".format(metric_names, sorted_metrics))
        sorted_metrics = metric_names
    return sorted_metrics


def plot_activations(model, samples, layer_name, mode='test', bins=100):
    """
    Make plots of activations of one layer of a model.

    Arrays will be flattend before plotting them as histograms.

    Parameters
    ----------
    model : keras model
        The model to make the data with.
    samples : dict
        Input data.
    layer_name : str or None
        Name of the layer to get info from. None for all layers.
    mode : str
        Mode of the layers during the forward pass. Either train or test.
        Important for batchnorm, dropout, ...
    bins : int
        Number of bins of the histogram.

    Returns
    -------
    fig : plt figure
        The plot of the activations of the given layer.

    """

    layer = model.get_layer(layer_name)
    if layer.name in model.input_names:
        activations = samples[layer.name]
    else:
        activations = get_layer_output(model, samples, layer.name, mode)

    fig, ax = plt.subplots()
    ax.hist(activations.flatten(), bins=bins)
    ax.set_title('Activations for layer ' + str(layer_name))
    ax.set_xlabel('Activation (layer output)')
    ax.set_ylabel('Quantity [#]')

    return fig


def plot_weights(model, layer_name, bins=100):
    """
    Make plots of the weights of one layer of a model.

    Arrays will be flattend before plotting them as histograms.

    Parameters
    ----------
    model : keras model
        The model to make the data with.
    layer_name : str or None
        Name of the layer to get info from. None for all layers.
    bins : int
        Number of bins of the histogram.

    Returns
    -------
    fig : plt figure or None
        The plot of the weights of the given layer. None if the layer
        has no weights.

    """
    layer = model.get_layer(layer_name)
    layer_weights = layer.get_weights()

    if not layer_weights:
        return None

    fig, ax = plt.subplots()

    # layer_weights is a list of np arrays; flatten it
    weights = np.array([])
    for j, w_temp in enumerate(layer_weights):
        w_temp_flattened = np.array(w_temp.flatten(), dtype=np.float64)
        weights = np.concatenate((weights, w_temp_flattened), axis=0)

    ax.hist(weights, bins=bins)
    ax.set_title('Weights for layer ' + str(layer_name))
    ax.set_xlabel('Weight')
    ax.set_ylabel('Quantity [#]')
    fig.tight_layout()

    return fig
