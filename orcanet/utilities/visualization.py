# -*- coding: utf-8 -*-
"""
Visualization tools used without Keras.
Makes performance graphs for training and validating.
"""
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class TrainValPlotter:
    """
    Class for plotting train/val curves.

    Instructions
    ------------
    1. Use tvp.plot_curves(train, val) once or more on pairs of
        train/val data.
    2. When all lines are plotted, use tvp.apply_layout() once for proper
        scaling, ylims, etc.

    """
    def __init__(self):
        # White space added below and above points
        self.y_lim_padding = [0.10, 0.25]
        # Store all plotted points for setting x/y lims
        self._xpoints_train = np.array([])
        self._xpoints_val = np.array([])
        self._ypoints_train = np.array([])
        self._ypoints_val = np.array([])

    def plot_curves(self,
                    train_data,
                    val_data=None,
                    train_label="training",
                    val_label="validation",
                    color=None,
                    smooth_sigma=None,
                    tlw=0.5,
                    vlw=0.5,
                    vms=3):
        """
        Plot a training and optionally a validation line.

        The data can contain nan's.

        Parameters
        ----------
        train_data : List
            X data [0] and y data [1] of the train curve. Will be plotted as
            connected dots.
        val_data : List, optional
            Optional X data [0] and y data [1] of the validation curve.
            Will be plotted as a faint solid line of the same color as train.
        train_label : str, optional
            Label for the train line in the legend.
        val_label : str, optional
            Label for the validation line in the legend.
        color : str, optional
            Color used for the train/val line.
        smooth_sigma : int, optional
            Apply gaussian blur to the train curve with given sigma.
        tlw : float
            Linewidth of train curve.
        vlw : float
            Linewidth of val curve.
        vms : float
            Markersize of the val curve.

        """
        if train_data is None and val_data is None:
            raise ValueError(
                "Can not plot when no train and val data is given.")

        if train_data is not None:
            epoch, y_data = train_data
            if smooth_sigma is not None:
                y_data = gaussian_smooth(y_data, smooth_sigma)

            self._xpoints_train = np.concatenate((self._xpoints_train, epoch))
            self._ypoints_train = np.concatenate((self._ypoints_train, y_data))

            train_plot = plt.plot(
                epoch, y_data, color=color, ls='-',
                zorder=3, label=train_label, lw=tlw, alpha=0.5)
            train_color = train_plot[0].get_color()
        else:
            train_color = color

        if val_data is not None:
            self._xpoints_val = np.concatenate((self._xpoints_val,
                                                val_data[0]))
            self._ypoints_val = np.concatenate((self._ypoints_val,
                                                val_data[1]))

            val_data_clean = skip_nans(val_data)
            # val plot always has the same color as the train plot
            plt.plot(val_data_clean[0], val_data_clean[1], color=train_color,
                     marker='o', zorder=3, lw=vlw, markersize=vms, label=val_label)

    def apply_layout(self,
                     title=None,
                     x_label="Epoch",
                     y_label=None,
                     grid=True,
                     legend=True,
                     x_lims=None,
                     y_lims="auto",
                     x_ticks="auto",
                     logy=False):
        """
        Apply given layout.
        Can calculate good y_lims and x_ticks automatically.

        Parameters
        ----------
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
        x_lims : List
            X limits of the data.
        y_lims : List or str
            Y limits of the data. "auto" for auto-calculation.
        x_ticks : List
            Positions of the major x ticks.
        logy : bool
            If true, make y axis log.

        """
        if logy:
            plt.yscale("log")
        if x_ticks is not None:
            if x_ticks == "auto":
                all_x_points = np.concatenate(
                    (self._xpoints_train, self._xpoints_val)
                )
                x_ticks = get_epoch_xticks(all_x_points)
            else:
                x_ticks = x_ticks
            plt.xticks(x_ticks)

        if x_lims is not None:
            plt.xlim(x_lims)

        if y_lims is not None:
            if y_lims == "auto":
                y_lims = get_ylims(
                    self._ypoints_train,
                    self._ypoints_val,
                    fraction=self.y_lim_padding,
                )
            else:
                y_lims = y_lims
            plt.ylim(y_lims)

        if legend:
            plt.legend(loc='upper right')

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if title is not None:
            title = plt.title(title)
            title.set_position([.5, 1.04])

        if grid:
            plt.grid(True, zorder=0, linestyle='dotted')


def gaussian_smooth(y, sigma, truncate=4):
    """ Smooth a 1d ndarray with a gaussian filter. """
    # kernel_width = 2 * sigma * truncate + 1
    kernel_x = np.arange(-truncate * sigma, truncate * sigma + 1)
    kernel = _gauss(kernel_x, 0, sigma)
    y = np.pad(np.asarray(y), int(len(kernel)/2), "edge")
    blurred = np.convolve(y, kernel, "valid")
    return blurred


def _gauss(x, mu=0, sigma=1):
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def plot_history(train_data,
                 val_data=None,
                 train_label="training",
                 val_label="validation",
                 color=None,
                 **kwargs):
    """
    Plot the train/val curves in a single plot.

    For backward compat. Functionality moved to TrainValPlotter

    """
    tvp = TrainValPlotter()
    tvp.plot_curves(train_data,
                    val_data,
                    train_label=train_label,
                    val_label=val_label,
                    color=color)
    tvp.apply_layout(**kwargs)


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


def get_ylims(y_points_train, y_points_val=None, fraction=0.25):
    """
    Get the y limits for the summary plot.

    For the training data, limits are calculated while ignoring data points
    which are far from the median (in terms of the median distance
    from the median).
    This is because there are outliers sometimes in the training data,
    especially early on in the training.

    Parameters
    ----------
    y_points_train : List
        y data of the train curve.
    y_points_val : List or None
        Y data of the validation curve.
    fraction : float or List
        How much whitespace of the total y range is added below and above
        the lines.

    Returns
    -------
    y_lims : tuple
        Minimum, maximum of the data.

    """
    assert not (y_points_train is None and
                y_points_val is None), "train and val data are None"

    def reject_outliers(data, threshold):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        no_outliers = data[s < threshold]
        lims = np.amin(no_outliers), np.amax(no_outliers)
        return lims

    mins, maxs = [], []
    if y_points_train is not None and len(y_points_train) != 0:
        y_train = y_points_train[~np.isnan(y_points_train)]
        y_lims_train = reject_outliers(y_train, 5)
        mins.append(y_lims_train[0])
        maxs.append(y_lims_train[1])

    if y_points_val is not None and len(y_points_val) != 0:
        y_val = y_points_val[~np.isnan(y_points_val)]

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

    try:
        fraction = float(fraction)
        padding = [fraction, fraction]
    except TypeError:
        # is a list
        padding = fraction

    if padding != [0., 0.]:
        y_lims = (y_lims[0] - padding[0] * y_range,  y_lims[1] + padding[1] * y_range)

    return y_lims


def get_epoch_xticks(x_points):
    """
    Calculates the xticks for the train and validation summary plot.

    One tick per epoch. Less the larger #epochs is.

    Parameters
    ----------
    x_points : List
        A list of the x coordinates of all points.

    Returns
    -------
    x_ticks_major : numpy.ndarray
        Array containing the ticks.

    """
    if len(x_points) == 0:
        raise ValueError("x-coordinates are empty!")

    minimum, maximum = np.amin(x_points), np.amax(x_points)
    if maximum - minimum > 0.5:
        # for longer trainings
        start_epoch, end_epoch = np.floor(minimum), np.ceil(maximum)
        # less xticks if there are many epochs
        n_epochs = end_epoch - start_epoch
        x_ticks_stepsize = 1 + np.floor(n_epochs / 20.)
        x_ticks_major = np.arange(
            start_epoch, end_epoch + x_ticks_stepsize, x_ticks_stepsize)
    else:
        # for early peeks
        start_epoch = np.floor(minimum)
        end_epoch = maximum + minimum - start_epoch
        x_ticks_major = np.linspace(
            start_epoch, end_epoch, 6)

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
    orga : orcanet.core.Organizer
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
            orga.history.plot_metric(
                metric, color=colors[color_counter % len(colors)])
            plt.suptitle(
                os.path.basename(os.path.abspath(orga.cfg.output_folder)))
            color_counter += 1
            pdf.savefig()
            plt.clf()

        orga.history.plot_lr()
        color_counter += 1
        pdf.savefig()
        plt.close()


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
