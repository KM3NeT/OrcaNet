"""
OrcaNet custom loss functions.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from orcanet.misc import get_register

# fuzz factor for numerical stability
EPS = tf.constant(1e-7, dtype="float32")
# for loading via toml and orcanet custom objects
loss_functions, _register = get_register()


@_register
def lkl_normal_tfp(y_true, y_pred):
    """ Normal distribution using tfp. See lkl_normal. """
    mu_true = y_true[:, 0]
    mu_pred, sigma_pred = y_pred[:, 0], y_pred[:, 1]

    return -1 * tfp.distributions.Normal(
        loc=mu_pred,
        scale=tf.math.maximum(sigma_pred, EPS),
    ).log_prob(mu_true)


@_register
def lkl_normal(y_true, y_pred):
    """
    Negative normal log-likelihood function for n regression output neurons
    with clipping for increased stability.

    For stability in the case of outliers, the loss l_i is capped
    at a maximum of 10 * |pred_i - true_i| for each sample.

    Parameters
    ----------
    y_true : tf.Tensor
        Shape (bs, 2, n) or (bs, 2).
        y_true[:, 0] is the label of shape (bs, n) (true), and y_true[:, 1]
        is not used (necessary as tf 2.1 requires y_true and y_pred to
        have same shape).
    y_pred : tf.Tensor
        Shape (bs, 2, n) or (bs, 2).
        The output of the network.
        y_pred[:, 0] is mu, and y_pred[:, 1] is sigma.

    """
    mu_true = y_true[:, 0]
    mu_pred, sigma_pred = y_pred[:, 0], y_pred[:, 1]

    return _normal_lkl(mu_pred=mu_pred, mu_true=mu_true, sigma_pred=sigma_pred, clip=True)


def _normal_lkl(mu_pred, mu_true, sigma_pred, clip=False, clip_thresh=10):
    delta = mu_pred - mu_true
    std_sq = sigma_pred**2
    loglike = tf.math.log(std_sq + EPS) + delta**2 / (std_sq + EPS)
    if clip:
        loglike = tf.minimum(loglike, clip_thresh*tf.abs(delta))
    return 0.5 * (tf.constant(np.log(2*np.pi), dtype="float32") + loglike)
