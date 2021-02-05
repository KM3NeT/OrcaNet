import orcanet.misc as misc

# for loading via toml
dmods, register = misc.get_register()


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
    datasets["pred"] = misc.dict_to_recarray(info_blob.get("y_pred"))

    ys = info_blob.get("ys")
    if ys is not None:
        datasets["true"] = misc.dict_to_recarray(ys)

    y_values = info_blob.get("y_values")
    if y_values is not None:
        datasets['y_values'] = y_values  # is already a structured array

    return datasets


@register
def as_recarray_dist(info_blob):
    """
    Save network output as recarray to h5. Intended for when network
    outputs are distributions and thus 3D (for example when using
    OutputRegNormal as output layer block).
    I.e. (batchsize, 2, X), with [:, 0] being mu and [:, 1] being std.

    Example output from network:
    shape {"A": (bs, 2), "B": (bs, 2, 3)}
        [:, 0] is reco, [:, 1] is err

    dtypes that will get saved to h5:
    A_1, A_err_1, B_1, B_2, B_3, B_err_1, B_err_2, B_err_3

    """
    y_pred = info_blob["y_pred"]
    datas = {}
    for output_name, array in y_pred.items():
        # [:, 0] is mu and [:, 1] is err
        datas[output_name] = array[:, 0]
        datas[f"{output_name}_err"] = array[:, 1]
    info_blob["y_pred"] = datas

    ys = info_blob.get("ys")
    if ys is not None:
        # errs for the trues are just padded, so skip
        datas = {}
        for output_name, array in ys.items():
            datas[output_name] = array[:, 0]
        info_blob["ys"] = datas

    return as_recarray(info_blob)


@register
def as_recarray_dist_split(info_blob):
    """
    Save network output as recarray to h5. Intended for networks that
    output recos and errs in seperate towers (for example when using
    OutputRegNormalSplit as output layer block).

    Example output from network:
    shape {"A": (bs, 1), "A_err": (bs, 2, 1),
           "B": (bs, 3), "B_err": (bs, 2, 3)}
    In "A_err": [:, 0] is mu, [:, 1] is sigma

    dtypes that will get saved to h5:
    A_1, A_err_1, B_1, B_1_err, B_2, B_err_2, ...

    """
    def transform(network_output):
        """ Skip A and rename A_err to A. """
        transformed = {}
        for output_name, output_value in network_output.items():
            if output_name.endswith("_err"):
                transformed[output_name[:-4]] = output_value
        return transformed

    info_blob["y_pred"] = transform(info_blob["y_pred"])
    if info_blob.get("ys") is not None:
        info_blob["ys"] = transform(info_blob["ys"])

    return as_recarray_dist(info_blob)
