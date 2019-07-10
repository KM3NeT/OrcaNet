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

    """

    layer = model.get_layer(layer_name)
    if layer.name in model.input_names:
        activations = samples[layer.name]
    else:
        activations = get_layer_output(model, samples, layer.name, mode)

    plt.hist(activations.flatten(), bins=bins)
    plt.title('Activations for layer ' + str(layer_name))
    plt.xlabel('Activation (layer output)')
    plt.ylabel('Quantity [#]')


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

    Raises
    ------
    ValueError
        If there are no weights in the given layer.

    """
    layer = model.get_layer(layer_name)
    layer_weights = layer.get_weights()

    if not layer_weights:
        raise ValueError("Layer contains no weights")

    # layer_weights is a list of np arrays; flatten it
    weights = np.array([])
    for j, w_temp in enumerate(layer_weights):
        w_temp_flattened = np.array(w_temp.flatten(), dtype=np.float64)
        weights = np.concatenate((weights, w_temp_flattened), axis=0)

    plt.hist(weights, bins=bins)
    plt.title('Weights for layer ' + str(layer_name))
    plt.xlabel('Weight')
    plt.ylabel('Quantity [#]')
    plt.tight_layout()

