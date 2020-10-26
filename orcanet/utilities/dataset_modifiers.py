import numpy as np
from orcanet.utilities.misc import get_register

# for loading via toml
dmods, register = get_register()


@register
def as_array(info_blob):
    """
    Save network output as ndarrays to h5. This is the default dataset modifier.

    Every output layer will get one dataset each for both the label and
    the prediction. E.g. if the model has an output layer called "energy",
    the datasets "label_energy" and "pred_energy" will be made.

    """
    datasets = dict()

    y_pred = info_blob["y_pred"]
    for out_layer_name in y_pred:
        datasets["pred_" + out_layer_name] = y_pred[out_layer_name]

    ys = info_blob.get("ys")
    if ys is not None:
        for out_layer_name in ys:
            datasets["label_" + out_layer_name] = ys[out_layer_name]

    y_values = info_blob.get("y_values")
    if y_values is not None:
        datasets['y_values'] = y_values

    return datasets


@register
def as_recarray(info_blob):
    """
    Save network output as recarray to h5. Intended for when network
    outputs are 2D, i.e. (batchsize, X).

    Output from network:
    Dict with arrays, shapes (batchsize, x_i).
    E.g. {"foo": ndarray, "bar": ndarray}

    dtypes that will get saved to h5:
    (foo_1, foo_2, ..., bar_1, bar_2, ... )

    """
    datasets = dict()
    datasets["pred"] = dict_to_recarray(info_blob.get("y_pred"))

    ys = info_blob.get("ys")
    if ys is not None:
        datasets["true"] = dict_to_recarray(ys)

    y_values = info_blob.get("y_values")
    if y_values is not None:
        datasets['y_values'] = y_values  # is already a structured array

    return datasets


def as_recarray_distr(info_blob):
    """
    Save network output as recarray to h5. Intended for when network
    outputs are distributions and thus 3D.
    I.e. (batchsize, 2, X), with [:, 0] being mu and [:, 1] being std.

    Output from network:
    shape {"A": (bs, 2), "B": (bs, 2, 3)}
        [:, 0] is reco, [:, 1] is err

    dtypes that will get saved to h5:
    A_1, A_err_1, B_1, B_1_err, B_2, B_err_2, ...

    """
    def output_to_rec(network_output, with_err=True):
        rec_dict = {}
        for output_name, output_array in network_output.items():
            if len(output_array.shape) == 2:
                output_array = np.expand_dims(output_array, -1)
            elif len(output_array.shape) != 3:
                raise ValueError(
                    f"Invalid shape {output_array.shape}: {output_array}")

            for i in range(output_array.shape[-1]):
                rec_dict[f"{output_name}_{i+1}"] = output_array[:, 0, i]
                if with_err:
                    rec_dict[f"{output_name}_err_{i+1}"] = output_array[:, 1, i]
        return utmisc.to_recarray(dictionary=rec_dict)

    datasets = dict()
    datasets["pred"] = output_to_rec(info_blob["y_pred"])

    ys = info_blob.get("ys")
    if ys is not None:
        # errs for the trues are just padded, so skip
        datasets["true"] = output_to_rec(ys, with_err=False)

    y_values = info_blob.get("y_values")
    if y_values is not None:
        datasets['y_values'] = y_values  # is already a structured array

    return datasets


def dict_to_recarray(array_dict):
    """
    Convert a dict with np arrays to a 2d recarray.
    Column names are derived from the dict keys.

    Parameters
    ----------
    array_dict : dict
        Keys: string
        Values: ND arrays, same length and number of dimensions.
            All dimensions expect first will get flattened.

    Returns
    -------
    ndarray
        The recarray.

    """
    column_names, arrays = [], []
    for key, array in array_dict.items():
        if len(array.shape) > 2:
            array = np.reshape(array, (len(array), -1))
        arrays.append(array)
        for i in range(array.shape[-1]):
            column_names.append(f"{key}_{i+1}")

    names = ",".join([name for name in column_names])
    data = np.concatenate(arrays, axis=1)
    return np.core.records.fromrecords(data, names=names)
