import keras.backend as K


def custom_metric_mean_relative_error_5_labels(y_true, y_pred):
    return K.abs(y_pred - y_true)/(y_true[0])

def loss_uncertainty_gaussian_likelihood(y_true, y_pred):
    y_pred_label = y_pred[0] # TODO PLACEHOLDER!!
    y_pred_label_std = y_pred[1] # TODO PLACEHOLDER!!
    y_true_label = y_true[0] # TODO PLACEHOLDER!!
    return K.log(y_pred_label_std ** 2) + (y_pred_label - y_true_label) ** 2 / (y_pred_label_std ** 2)



