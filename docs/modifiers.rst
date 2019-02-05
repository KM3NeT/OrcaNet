Modifiers in OrcaNet
====================
.. contents:: :local:
Modifiers are used to preprocess data from the input files, before handing them to the network.
They are functions, and can be applied to both the samples, as well as the labels.
They required the input and output layers of the network, as well as ever input set in the toml list file to be named.
This makes it easy to assure that the right data is fed into the right layer of the network, especially if you have multiple inputs or outputs.

Sample modifier
---------------
Sample modifiers (``cfg.sample_modifier``) are used to distribute the samples read from the h5 input file to the right input layer in the network.
They must have the following form:

.. code-block:: python
    def sample_modifier(xs_list):
        ...
        return xs_layer

``xs_list``: ``dict``
    Toml list input set names as keys, one batch of data as values.
``xs_layer``: ``dict``
    Model input layer names as keys, one batch of data as values.

Hint: If the names of the input sets in the toml list file and the names of the input layers match, no sample modifier is required!


Example
^^^^^^^
A simple model with two inputs.
We have xy and yz projections in the input files, but want to feed the network xy and zy data.
Content of the toml list file:

.. code-block:: toml
    [xy]
    train_files = [
    "data/xy_train.h5",
    ]

    validation_files = [
    "data/xy_val.h5"
    ]

    [yz]
    train_files = [
    "data/yz_train.h5",
    ]

    validation_files = [
    "data/yz_val.h5"
    ]

Used keras model:

.. code-block:: python
    inp_1 = Input((1,), name="input_layer_xy")
    inp_2 = Input((1,), name="input_layer_zy")

    x = Concatenate()([inp_1, inp_2])
    output = Dense(10)(x)

    test_model = Model((inp_1, inp_2), output)

The following sample modifier is required:

.. code-block:: python
    def sample_modifier(xs_list):
        xs_layer = dict()
        xs_layer["input_layer_xy"] = xs_list["xy"]
        yz_data = xs_list["yz"]
        xs_layer["input_layer_zy"] = np.swapaxes(yz_data, 1, 2)  # Axis 0 is the batchsize!
        return xs_layer


Label modifier
--------------
Label modifiers (``cfg.label_modifier``) are used to generate labels for the model from the mc_info of h5 input files.
They must have the following form:

.. code-block:: python
    def label_modifier(mc_info):
        ...
        return y_true

``mc_info``: ``numpy structured array``
    One batch read from the h5 input files. Contains all the info for each sample in the batch with the name of each property as a dtype name.
``y_true``: ``dict``
    Model output layer names as keys, one batch of labels as values.

Hint: If the names of the dtypes in the toml list file contain the names of the output layers, no sample modifier is required!


Dataset modifier
----------------
Dataset modifiers (``cfg.dataset_modifier``) are used by when a model is evaluated with ``orca_eval``.
They will determine what is written in which datasets in the resulting evaluation h5 file.
They must have the following form:

.. code-block:: python
    def dataset_modifier(mc_info, y_true, y_pred)
        ...
        return datasets

``mc_info``: ``numpy structured array``
    One batch read from the h5 input files. Contains all the info for each sample in the batch with the name of each property as a dtype name.
``y_true``: ``dict``
    Model output layer names as keys, one batch of labels as values.
``y_pred``: ``dict``
    Model output layer names as keys, one batch of predictions of the network as values.
``datasets``: ``dict``
    Every key determines a dataset to be created in the h5 file. The values are what will be the content of each dataset.

Hint: If no dataset modifier is given, the following datasets will be created: mc_info, and two sets for every output layer (label and pred).

