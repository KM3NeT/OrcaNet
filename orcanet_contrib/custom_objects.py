import tensorflow.keras.backend as K
import tensorflow as tf
import math


def get_custom_objects():
    """
    Functions that returns a dict with all relevant loss functions in this file.
    """
    custom_objects = {
        'loss_direction': loss_direction,
        'loss_uncertainty_mse': loss_uncertainty_mse,
        'loss_uncertainty_mae': loss_uncertainty_mae,
        'loss_uncertainty_gaussian_likelihood': loss_uncertainty_gaussian_likelihood,
        'loss_uncertainty_gaussian_likelihood_dir': loss_uncertainty_gaussian_likelihood_dir,
        'loss_mean_relative_error_energy': loss_mean_relative_error_energy
    }
    return custom_objects


def loss_mean_relative_error_energy(y_true, y_pred):
    """
    Loss function that calculates the mean relative error.
    y_true & y_pred are expected to be e_true & e_pred.
    L = (e_reco - e_true) / e_true

    Returns
    -------
    mre : Mean relative (energy) error loss

    """
    mre = K.abs(y_pred - y_true)/y_true
    return mre


def loss_uncertainty_mae(y_true, y_pred):
    """
    Mean absolute error loss for the uncertainty estimation.
    L = sigma_pred / abs(label_true - label_reco).

    Returns
    -------
    loss : Mean absolute error for uncertainty estimations.

    """
    # order in y_pred: 1) pred label 2) pred label error
    # prevent that the gradient flows back over the label network:
    y_pred_label = K.stop_gradient(y_pred[:, 0])
    y_pred_label_std = y_pred[:, 1]
    y_true_label = y_true[:, 0]

    # (s - |y_true - y_pred|)
    loss = K.abs(y_pred_label_std - K.abs(y_true_label - y_pred_label))
    return loss


def loss_uncertainty_mse(y_true, y_pred):
    """
    Mean squared error loss for the uncertainty estimation.
    L = sigma_pred / abs(label_true - label_reco)**2.

    Returns
    -------
    loss : Mean squared error for uncertainty estimations.

    """
    # order in y_pred: 1) pred label 2) pred label error
    # prevent that the gradient flows back over the label network:
    y_pred_label = K.stop_gradient(y_pred[:, 0])
    y_pred_label_std = y_pred[:, 1]
    y_true_label = y_true[:, 0]

    # (s - |y_true - y_pred|)**2
    loss = K.pow(y_pred_label_std - K.abs(y_true_label - y_pred_label), 2)
    return loss


def loss_uncertainty_gaussian_likelihood(y_true, y_pred):
    """
    Loss function that calculates something similar to a Gaussian Likelihood.
    Requires that y_pred contains only one predicted value (label).
    y_true & y_pred are expected to contain the predicted/true label and
    the predicted std for the label.
    L = ln(std ** 2) + (y_label_pred - y_label_true) / (std ** 2)

    Returns
    -------
    loss : Gaussian Likelihood loss

    """
    # order in y_pred: 1) pred label 2) pred label error
    # prevent that the gradient flows back over the label network:
    y_pred_label = K.stop_gradient(y_pred[:, 0])
    y_pred_label_std = y_pred[:, 1]
    y_true_label = y_true[:, 0]

    # equal to a lower std limit of 3.16 e-2
    eps = tf.constant(1e-3, dtype="float32")
    # y_pred_label_std += eps

    loss = K.log(K.pow(y_pred_label_std, 2) + eps) + K.pow(y_pred_label - y_true_label, 2) / (K.pow(y_pred_label_std, 2) + eps)
    return loss


def loss_uncertainty_gaussian_likelihood_dir(y_true, y_pred):
    """
    Loss function that calculates something similar to a Gaussian
    Likelihood for predicted directions. Requires that y_pred contains
    three predicted values (labels): dir_x, dir_y, dir_z.
    y_true & y_pred are expected to contain the predicted/true label and
    the predicted std for the label.
    L = ln(std ** 2) + (y_label_pred - y_label_true) / (std ** 2)

    Returns
    -------
    loss : Gaussian Likelihood loss for the directional error

    """
    # order in y_pred: 1) pred label 2) pred label error
    # prevent that the gradient flows back over the label network
    y_pred_dir_x, y_pred_dir_y, y_pred_dir_z = K.stop_gradient(y_pred[:, 0]), K.stop_gradient(y_pred[:, 1]), K.stop_gradient(y_pred[:, 2])
    y_pred_std_dir_x, y_pred_std_dir_y, y_pred_std_dir_z = y_pred[:, 3], y_pred[:, 4], y_pred[:, 5]
    y_true_dir_x, y_true_dir_y, y_true_dir_z = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    # equal to a lower std limit of 1e-3
    eps = tf.constant(1e-6, dtype="float32")

    loss_dir_x = K.log(K.pow(y_pred_std_dir_x, 2) + eps) + K.pow(y_pred_dir_x - y_true_dir_x, 2) / (K.pow(y_pred_std_dir_x, 2) + eps)
    loss_dir_y = K.log(K.pow(y_pred_std_dir_y, 2) + eps) + K.pow(y_pred_dir_y - y_true_dir_y, 2) / (K.pow(y_pred_std_dir_y, 2) + eps)
    loss_dir_z = K.log(K.pow(y_pred_std_dir_z, 2) + eps) + K.pow(y_pred_dir_z - y_true_dir_z, 2) / (K.pow(y_pred_std_dir_z, 2) + eps)

    loss = loss_dir_x + loss_dir_y + loss_dir_z
    return loss


def loss_direction(y_true, y_pred):
    """
    Loss function that calculates the space angle between the predicted
    and the true direction.
    Not used anymore, can lead to inf gradients due to tf.acos(space_angle_inner_value)!
    Converts cartesian dirs to spherical coordinate system and then
    calculates the space angle between the two vectors.

    Returns
    -------
    total_loss : Space angle loss
    """
    # define dir_preds
    y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=300)
    y_pred_x, y_pred_y, y_pred_z = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    y_true_x, y_true_y, y_true_z = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    # convert cartesian coordinates to spherical coordinates
    r_pred = K.sqrt(K.pow(y_pred_x, 2) + K.pow(y_pred_y, 2) + K.pow(y_pred_z, 2))
    # y_true input should be normalized! can also use 1 instead of this:
    r_true = K.sqrt(K.pow(y_true_x, 2) + K.pow(y_true_y, 2) + K.pow(y_true_z, 2))

    eps = tf.constant(1e-7, dtype="float32")  # TODO test

    # alternatively zenith_pred = tf.acos(y_pred_z / r_pred + K.epsilon()):
    zenith_pred = tf.atan2(y_pred_z + eps, K.sqrt(K.pow(y_pred_x, 2) + K.pow(y_pred_y, 2)) + eps)
    zenith_true = tf.atan2(y_true_z + eps, K.sqrt(K.pow(y_true_x, 2) + K.pow(y_true_y, 2)) + eps)
    azimuth_pred, azimuth_true = tf.atan2(y_pred_y + eps, y_pred_x + eps), tf.atan2(y_true_y + eps, y_true_x + eps)

    # shift azimuth and zenith by pi / pi/2 in order to make the space angle formula work
    pi = math.pi
    zenith_pred, zenith_true = zenith_pred + tf.constant(pi/float(2), dtype="float32"), zenith_true + tf.constant(pi/float(2), dtype="float32")
    azimuth_pred, azimuth_true = azimuth_pred + tf.constant(pi, dtype="float32"), azimuth_true + tf.constant(pi, dtype="float32")

    # calculate space angle between the two vectors (true, pred) in spherical coordinates, cf. bachelor thesis shallmann
    # protect space angle from acos values outside of [-1,1] range
    space_angle_inner_value = tf.sin(zenith_true) * tf.sin(zenith_pred) * tf.cos(azimuth_true - azimuth_pred) + tf.cos(zenith_true) * tf.cos(zenith_pred)
    space_angle_inner_value = K.clip(space_angle_inner_value, -1, 1)

    space_angle = tf.acos(space_angle_inner_value)
    loss_r = K.abs(r_true - r_pred)

    total_loss = loss_r + space_angle
    # total_loss = tf.Print(total_loss, [total_loss], message='total_loss', summarize=64)
    return total_loss


def loss_direction_grad():
    """
    Similar loss function as in loss_direction, but here it is just used
    for manually calculating the gradient in the  main function. Done in
    order to protect the loss_direction function from inf gradients.

    """
    # define dir_preds
    y_true = np.array([[1, 0, 0], [0, 0, 1]])
    y_true = tf.constant(y_true, shape=y_true.shape, dtype='float32')
    y_pred = np.array([[-1, 0, 0], [0, 0, -1]])
    y_pred = tf.Variable(y_pred, dtype='float32')
    y_pred_x, y_pred_y, y_pred_z = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    y_true_x, y_true_y, y_true_z = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    eps = tf.constant(1e-7, dtype="float32")  # TODO test

    # alternatively zenith_pred = tf.acos(y_pred_z / r_pred + K.epsilon()):
    zenith_pred = tf.atan2(y_pred_z + eps, K.sqrt(K.pow(y_pred_x, 2) + K.pow(y_pred_y, 2)) + eps)
    zenith_true = tf.atan2(y_true_z + eps, K.sqrt(K.pow(y_true_x, 2) + K.pow(y_true_y, 2)) + eps)
    azimuth_pred, azimuth_true = tf.atan2(y_pred_y + eps, y_pred_x + eps), tf.atan2(y_true_y + eps, y_true_x + eps)

    # shift azimuth and zenith by pi / pi/2 in order to make the space angle formula work
    pi = math.pi
    zenith_pred, zenith_true = zenith_pred + tf.constant(pi/float(2), dtype="float32"), zenith_true + tf.constant(pi/float(2), dtype="float32")
    azimuth_pred, azimuth_true = azimuth_pred + tf.constant(pi, dtype="float32"), azimuth_true + tf.constant(pi, dtype="float32")

    # calculate space angle between the two vectors (true, pred) in spherical coordinates, cf. bachelor thesis shallmann
    # protect space angle from acos values outside of [-1,1] range
    space_angle_inner_value = tf.sin(zenith_true) * tf.sin(zenith_pred) * tf.cos(azimuth_true - azimuth_pred) + tf.cos(zenith_true) * tf.cos(zenith_pred)
    space_angle_inner_value = K.clip(space_angle_inner_value, -1, 1)
    space_angle = tf.acos(space_angle_inner_value)

    return space_angle, y_pred


def mean_absolute_error(y_true, y_pred):
    """
    Copy of the Keras mean absolute error function for testing purposes.
    """
    # y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=5)
    # y_true = tf.Print(y_true, [y_true], message='y_true', summarize=5)
    absolute = K.abs(y_pred - y_true)
    # absolute = tf.Print(absolute, [absolute], message='absolute', summarize=5)
    mae = K.mean(absolute, axis=-1)
    return mae


if __name__ == "__main__":
    import numpy as np
    with tf.Session() as sess:
        dir_y_true = np.array([[0.4, 0.1, 0.5], [0.2, -0.4, 0.4], [1, 1, 1], [0.4, 0.2, 0.2], [0.1, 0.2, 0.5]])
        dir_y_pred = np.array([[0.35, 0.05, 0.6], [0.15, -0.3, 0.55], [-1, -1, -1], [-0.4, -0.2, -0.2], [0, 0, 0]])
        x = tf.constant(dir_y_true, shape=dir_y_true.shape, dtype="float32")
        y = tf.constant(dir_y_pred, shape=dir_y_pred.shape, dtype="float32")
        # print('Final loss dir: ' + str(sess.run(loss_direction(x,y))))
        print('Final loss dir: ' + str(sess.run(mean_absolute_error(x, y))))

        # calc gradients for loss_direction
        space_angle_loss_func, y_pred_p = loss_direction_grad()
        init = tf.global_variables_initializer()
        sess.run(init)
        print('Final loss gradients dir: ' + str(sess.run(tf.gradients(space_angle_loss_func, y_pred_p))))

        # errors
        energy_y_true = np.array([[30], [6.35]])
        energy_and_std_y_pred = np.array([[25, 5], [5.84, 0.5]])
        x = tf.constant(energy_y_true, shape=energy_y_true.shape, dtype="float32")
        y = tf.constant(energy_and_std_y_pred, shape=energy_and_std_y_pred.shape, dtype="float32")
        print('Final loss energy uncertainty: ' + str(sess.run(loss_uncertainty_gaussian_likelihood(x, y))))

        dir_y_true = np.array([[0.4, 0.1, 0.5], [0.2, -0.4, 0.4], [1, 1, 1]])
        dir_and_std_y_pred = np.array([[0.35, 0.05, 0.6, 0.05, 0.05, 0.05], [0.15, -0.3, 0.55, 0.05, 0.1, 0.15], [-1, -1, -1, 2, 2, 2]])
        x = tf.constant(dir_y_true, shape=dir_y_true.shape, dtype="float32")
        y = tf.constant(dir_and_std_y_pred, shape=dir_and_std_y_pred.shape, dtype="float32")
        print('Final loss dir uncertainty: ' + str(sess.run(loss_uncertainty_gaussian_likelihood_dir(x, y))))
