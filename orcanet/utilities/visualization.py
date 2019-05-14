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


def plot_history(
        train_data, val_data=None,
        color=None, title=None, x_label="Epoch", y_label=None, grid=True,
        legend=True, train_label="training", val_label="validation",
        x_lims=None, y_lims="auto", x_ticks=None):
    """
    Plot a training and optionally a validation line in a single plot.

    The val data can contain nan's.

    Parameters
    ----------
    train_data : List
        X data [0] and y data [1] of the train curve. Will be plotted as
        connected dots.
    val_data : List
        Optional X data [0] and y data [1] of the validation curve.
        Will be plotted as a faint solid line of the same color as train.
    color : str
        Color used for the train/val line.
    title : str
        Title of the plot.
    x_label : str
        X label of the plot.
    y_label : str
        Y label of the plot.
    grid : bool
        If true, show a grid.
    legend : bool
        If true, show a legend.
    train_label : str
        Label for the train line in the legend.
    val_label : str
        Label for the validation line in the legend.
    x_lims : List
        X limits of the data.
    y_lims : List or str
        Y limits of the data. "auto" for auto-calculation.
    x_ticks : List
        Positions of the major x ticks.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot.

    """
    fig, ax = plt.subplots()
    plot_curves(train_data, val_data,
                train_label=train_label, val_label=val_label, color=color)

    if x_ticks is None:
        x_ticks = get_epoch_xticks(train_data, val_data)
    plt.xticks(x_ticks)

    if x_lims is not None:
        plt.xlim(x_lims)

    if y_lims is not None:
        if y_lims == "auto":
            y_lims = get_ylims(train_data, val_data, fraction=0.25)
        plt.ylim(y_lims)

    if legend:
        ax.legend(loc='upper right')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title is not None:
        title = plt.title(title)
        title.set_position([.5, 1.04])

    if grid:
        plt.grid(True, zorder=0, linestyle='dotted')

    return fig


def plot_curves(train_data, val_data=None,
                train_label="training", val_label="validation", color=None,
                train_smooth_ksize=None):
    """
    Plot a training and validation line.

    Parameters
    ----------
    train_data : List
        X data [0] and y data [1] of the train curve. Will be plotted as
        connected dots.
    val_data : List, optional
        Optional X data [0] and y data [1] of the validation curve.
        Will be plotted as a faint solid line of the same color as train.
    color : str, optional
        Color used for the train/val line.
    train_label : str, optional
        Label for the train line in the legend.
    val_label : str, optional
        Label for the validation line in the legend.
    train_smooth_ksize : int, optional
        Smooth the train curve by averaging over a moving window of given size.

    """
    if train_data is None and val_data is None:
        raise ValueError("Can not plot when no train and val data is given.")

    if train_data is not None:
        if train_smooth_ksize is not None:
            kernel = np.ones(train_smooth_ksize)/train_smooth_ksize
            epoch = np.convolve(train_data[0], kernel, 'valid')
            y_data = np.convolve(train_data[1], kernel, 'valid')
        else:
            epoch, y_data = train_data

        train_plot = plt.plot(
            epoch, y_data, color=color, ls='-',
            zorder=3, label=train_label, lw=0.6, alpha=0.5)
        train_color = train_plot[0].get_color()
    else:
        train_color = color

    if val_data is not None:
        val_data_clean = skip_nans(val_data)
        # val plot always has the same color as the train plot
        plt.plot(val_data_clean[0], val_data_clean[1], color=train_color,
                 marker='o', zorder=3, label=val_label)


def skip_nans(data):
    """
    Skip over nan values, so that all dots are connected.

    Parameters
    ----------
    data : List
        Contains x and y data as ndarrays. The y values may contain nans.

    Returns
    -------
    data_clean : List
        Contains x and y data as ndarrays. Points with y=nan are skipped.

    """
    not_nan = ~np.isnan(data[1])
    data_clean = data[0][not_nan], data[1][not_nan]
    return data_clean


def get_ylims(train_data, val_data=None, fraction=0.25):
    """
    Get the y limits for the summary plot.

    For the training data, limits are calculated while ignoring data points
    which are far from the median (in terms of the median distance
    from the median).
    This is because there are outliers sometimes in the training data,
    especially early on in the training.

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
    assert not (train_data is None and
                val_data is None), "train and val data are None"

    def reject_outliers(data, threshold):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        no_outliers = data[s < threshold]
        lims = np.amin(no_outliers), np.amax(no_outliers)
        return lims

    mins, maxs = [], []
    if train_data is not None:
        y_train = skip_nans(train_data)[1]
        y_lims_train = reject_outliers(y_train, 5)
        mins.append(y_lims_train[0])
        maxs.append(y_lims_train[1])

    if val_data is not None:
        y_val = skip_nans(val_data)[1]

        if len(y_val) == 1:
            y_lim_val = y_val[0], y_val[0]
        else:
            y_lim_val = np.amin(y_val), np.amax(y_val)

        mins.append(y_lim_val[0])
        maxs.append(y_lim_val[1])

    if len(mins) == 1:
        y_lims = (mins[0], maxs[0])
    else:
        y_lims = np.amin(mins), np.amax(maxs)

    if y_lims[0] == y_lims[1]:
        y_range = 0.1 * y_lims[0]
    else:
        y_range = y_lims[1] - y_lims[0]

    if fraction != 0:
        y_lims = (y_lims[0] - fraction * y_range,  y_lims[1] + fraction * y_range)

    return y_lims


def get_epoch_xticks(train_data, val_data=None):
    """
    Calculates the xticks for the train and validation summary plot.

    One tick per poch. Less the larger #epochs is.

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
    assert not (train_data is None and
                val_data is None), "train and val data are None"
    x_points = np.array([])

    if train_data is not None:
        x_points = np.append(x_points, train_data[0])
    if val_data is not None:
        x_points = np.append(x_points, val_data[0])

    minimum, maximum = np.amin(x_points), np.amax(x_points)
    start_epoch, end_epoch = np.floor(minimum), np.ceil(maximum)
    # reduce number of x_ticks by factor of 2 if n_epochs > 20
    n_epochs = end_epoch - start_epoch
    x_ticks_stepsize = 1 + np.floor(n_epochs / 20.)
    x_ticks_major = np.arange(
        start_epoch, end_epoch + x_ticks_stepsize, x_ticks_stepsize)

    return x_ticks_major


def update_summary_plot(orga):
    """
    Plot and save all metrics of the given validation- and train-data
    into a pdf file, each metric in its own plot.

    If metric pairs of a variable and its error are found (e.g. e_loss
    and e_err_loss), they will have the same color and appear back to
    back in the plot.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.

    """
    plt.ioff()
    pdf_name = orga.io.get_subfolder("plots", create=True) + "/summary_plot.pdf"

    # Extract the names of the metrics
    all_metrics = orga.history.get_metrics()
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
            fig = orga.history.plot_metric(
                metric, color=colors[color_counter % len(colors)])
            color_counter += 1
            pdf.savefig(fig)
            plt.close(fig)

        lr_fig = orga.history.plot_lr()
        color_counter += 1
        pdf.savefig(lr_fig)
        plt.close(lr_fig)


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

    assert 0 not in sorted_metrics, "Something went wrong with the sorting of " \
                                    "metrics! Given was {}, output was " \
                                    "{}. ".format(metric_names, sorted_metrics)

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
