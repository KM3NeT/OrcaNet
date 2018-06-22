import keras.backend as K


def custom_metric_mean_relative_error_5_labels(y_true, y_pred):
    return K.abs(y_pred - y_true)/(y_true[0])



