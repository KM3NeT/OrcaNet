import keras.backend as K
import tensorflow as tf
import math
import inspect
import sys


def get_all_loss_functions():
    loss_functions = iter(inspect.getmembers(sys.modules[__name__], inspect.isfunction)) # contains ['loss_func_name', loss_func, 'loss_func_2_name', ...]
    custom_objects = {}
    for loss_func_name in loss_functions:
        custom_objects[loss_func_name] = next(loss_functions)

    return custom_objects


def mean_relative_error_energy(y_true, y_pred):
    return K.abs(y_pred - y_true)/(y_true[:, 0])


def loss_uncertainty_gaussian_likelihood(y_true, y_pred):
    # TODO protect negative log, maybe use relu for sigma output?
    # order in y_pred: 1) pred label 2) pred label error
    y_pred_label = K.stop_gradient(y_pred[:, 0]) # prevent that the gradient flows back over the label network
    y_pred_label_std = y_pred[:, 1]
    y_true_label = y_true[:, 0]

    # y_pred_label = tf.Print(y_pred_label, [y_pred_label], message='y_pred_label: ')
    # y_pred_label_std = tf.Print(y_pred_label_std, [y_pred_label_std], message='y_pred_label_std: ')
    # y_true_label = tf.Print(y_true_label, [y_true_label], message='y_true_label: ')

    eps = tf.constant(1e-3, dtype="float32")
    y_pred_label_std += eps

    loss = K.log(K.pow(y_pred_label_std, 2)) + K.pow(y_pred_label - y_true_label, 2) / (K.pow(y_pred_label_std, 2))
    return loss / 1e6


def loss_uncertainty_gaussian_likelihood_dir(y_true, y_pred):
    # order in y_pred: 1) pred label 2) pred label error
    # prevent that the gradient flows back over the label network
    y_pred_dir_x, y_pred_dir_y, y_pred_dir_z = K.stop_gradient(y_pred[:, 0]), K.stop_gradient(y_pred[:, 1]), K.stop_gradient(y_pred[:, 2])
    y_pred_std_dir_x, y_pred_std_dir_y, y_pred_std_dir_z = y_pred[:, 3], y_pred[:, 4], y_pred[:, 5]
    y_true_dir_x, y_true_dir_y, y_true_dir_z = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    # y_pred_std_dir_x = tf.Print(y_pred_std_dir_x, [y_pred_std_dir_x], message='y_pred_std_dir_x: ')
    # y_pred_std_dir_y = tf.Print(y_pred_std_dir_y, [y_pred_std_dir_y], message='y_pred_std_dir_y: ')
    # y_pred_std_dir_z = tf.Print(y_pred_std_dir_z, [y_pred_std_dir_z], message='y_pred_std_dir_z: ')

    eps = tf.constant(1e-3, dtype="float32")
    y_pred_std_dir_x += eps
    y_pred_std_dir_y += eps
    y_pred_std_dir_z += eps

    loss_dir_x = K.log(K.pow(y_pred_std_dir_x, 2)) + K.pow(y_pred_dir_x - y_true_dir_x, 2) / (K.pow(y_pred_std_dir_x, 2))
    loss_dir_y = K.log(K.pow(y_pred_std_dir_y, 2)) + K.pow(y_pred_dir_y - y_true_dir_y, 2) / (K.pow(y_pred_std_dir_y, 2))
    loss_dir_z = K.log(K.pow(y_pred_std_dir_z, 2)) + K.pow(y_pred_dir_z - y_true_dir_z, 2) / (K.pow(y_pred_std_dir_z, 2))

    loss = loss_dir_x + loss_dir_y + loss_dir_z
    return loss / 1e6


def loss_direction(y_true, y_pred):
    # todo check division by zero
    # TODO norm direction
    # define dir_preds
    y_pred_x, y_pred_y, y_pred_z = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    y_true_x, y_true_y, y_true_z = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    # convert cartesian coordinates to spherical coordinates
    r_pred= K.sqrt(K.pow(y_pred_x, 2) + K.pow(y_pred_y, 2) + K.pow(y_pred_z, 2))
    r_true = K.sqrt(K.pow(y_true_x, 2) + K.pow(y_true_y, 2) + K.pow(y_true_z, 2))  # y_true input should be normalized! can also use 1 instead of this

    # r_pred = tf.Print(r_pred, [r_pred], message='r_pred: ', summarize=10)
    # y_pred_z = tf.Print(y_pred_z, [y_pred_z], message='y_pred_z', summarize=10)

    zenith_pred = tf.atan2(y_pred_z, K.sqrt(K.pow(y_pred_x, 2) + K.pow(y_pred_y, 2))) # alternatively zenith_pred = tf.acos(y_pred_z / r_pred + K.epsilon())
    zenith_true = tf.atan2(y_true_z, K.sqrt(K.pow(y_true_x, 2) + K.pow(y_true_y, 2)))
    azimuth_pred, azimuth_true = tf.atan2(y_pred_y, y_pred_x), tf.atan2(y_true_y, y_true_x)

    # shift azimuth and zenith by pi / pi/2 in order to make the space angle formula work
    zenith_pred = zenith_pred + tf.constant(math.pi/float(2), dtype="float32")
    zenith_true = zenith_true + tf.constant(math.pi/float(2), dtype="float32")
    azimuth_pred = azimuth_pred + tf.constant(math.pi, dtype="float32")
    azimuth_true = azimuth_true + tf.constant(math.pi, dtype="float32")

    # zenith_pred = tf.Print(zenith_pred, [zenith_pred], message='zenith_pred: ', summarize=10)
    # zenith_true = tf.Print(zenith_true, [zenith_true], message='zenith_true: ', summarize=10)
    # azimuth_pred = tf.Print(azimuth_pred, [azimuth_pred], message='azimuth_pred: ', summarize=10)
    # azimuth_true = tf.Print(azimuth_true, [azimuth_true], message='azimuth_true: ', summarize=10)

    # calculate space angle between the two vectors (true, pred) in spherical coordinates, cf. bachelor thesis shallmann
    # protect space angle from acos values outside of [-1,1] range
    space_angle_inner_value = tf.sin(zenith_true) * tf.sin(zenith_pred) * tf.cos(azimuth_true - azimuth_pred) \
                                + tf.cos(zenith_true) * tf.cos(zenith_pred)
    space_angle_inner_value = K.clip(space_angle_inner_value, -1, 1) # TODO check if this really only happens for numerical precision and not due to some bugs

    space_angle = tf.acos(space_angle_inner_value)

    # space_angle = tf.Print(space_angle, [space_angle], message='space_angle: ', summarize=10)

    loss_r = K.abs(r_true - r_pred)
    # loss_r = tf.Print(loss_r, [loss_r], message='loss_r', summarize=10)

    total_loss = loss_r + space_angle
    return total_loss / 2 # divide by 2 to be smaller than energy loss


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__=="__main__":
    import numpy as np
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        dir_y_true = np.array([[0.4, 0.1, 0.5], [0.2, -0.4, 0.4], [1,1,1], [0.4, 0.2, 0.2]])
        dir_y_pred = np.array([[0.35, 0.05, 0.6], [0.15, -0.3, 0.55], [-1,-1,-1], [-0.4, -0.2, -0.2]])
        x = tf.constant(dir_y_true, shape=dir_y_true.shape, dtype="float32")
        y = tf.constant(dir_y_pred, shape=dir_y_pred.shape, dtype="float32")
        print('Final loss dir: ' + str(sess.run(loss_direction(x,y))))

        # errors
        energy_y_true = np.array([[30], [6.35]])
        energy_and_std_y_pred = np.array([[25, 5], [5.84, 0.5]])
        x = tf.constant(energy_y_true, shape=energy_y_true.shape, dtype="float32")
        y = tf.constant(energy_and_std_y_pred, shape=energy_and_std_y_pred.shape, dtype="float32")
        print('Final loss energy uncertainty: ' + str(sess.run(loss_uncertainty_gaussian_likelihood(x, y))))

        dir_y_true = np.array([[0.4, 0.1, 0.5], [0.2, -0.4, 0.4], [1,1,1]])
        dir_and_std_y_pred = np.array([[0.35, 0.05, 0.6, 0.05, 0.05, 0.05], [0.15, -0.3, 0.55, 0.05, 0.1, 0.15], [-1,-1,-1, 2,2,2]])
        x = tf.constant(dir_y_true, shape=dir_y_true.shape, dtype="float32")
        y = tf.constant(dir_and_std_y_pred, shape=dir_and_std_y_pred.shape, dtype="float32")
        print('Final loss dir uncertainty: ' + str(sess.run(loss_uncertainty_gaussian_likelihood_dir(x, y))))




