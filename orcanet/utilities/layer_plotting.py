"""
For plotting keras and model related stuff.
"""

import numpy as np
import matplotlib.pyplot as plt

from orcanet.utilities.nn_utilities import get_layer_output


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
