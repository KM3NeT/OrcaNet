.. _modifiers_page:

Modifiers
=========
.. contents:: Modifier types:
    :local:
    :depth: 1

Modifiers are used to preprocess data from the input files, before handing them
to the network.
They are functions which are applied to both the samples, as well as the labels.
They require the input and output layers of the network, as well as ever input
set in the toml list file to be named.
This makes it easy to assure that the right data is fed into the right layer of
the network, especially if there are multiple inputs or outputs.

All modifiers take a dict ``info_blob`` as input, which contains a subset of
the following keys:

**Possible keys in the info_blob dict**
    ``x_values`` : dict
        One batch of data from the ``cfg.key_x_values`` datagroup of the h5 file.

        Keys: Input set names from the toml list file.

        Values: Numpy array with x values from the respective file.
        If the datagroup is an indexed dataset, this will be a tuple of numpy arrays instead,
        with [0] being the values, and [1] being the number of
        items per sample.
    ``y_values`` : ndarray
        One batch of data from the ``cfg.key_y_values`` datagroup of the h5 file.
        If the content of the datagroup is a structured array, this will
        also be a structured array.
    ``phase`` : str
        Current phase the network is used in. Either 'training', 'validation'
        or 'inference'. Can be used to have modifiers with different behaviours
        depending on the phase.
    ``xs`` : dict
        One batch of data, resulting from applying the sample modifier on ``x_values``.

        Keys: Name of an input layer of the network.

        Values: Numpy array with samples for the respective layer.
    ``ys`` : dict
        One batch of data, resulting from applying the label modifier on ``y_values``, aka
        the true values the model will try to reproduce.

        Keys: The names of the output layers of the model.

        Values: One batch of labels as a numpy array.
    ``y_pred`` : dict
        One batch of data, resulting from applying the model on ``xs``, aka the
        model prediction for this batch of input data.

        Keys: The names of the output layers of the model.

        Values: One batch of predictions from the respective output layer of the
        model as a numpy array.


For each modifier, a different subset of these entries will be available in the
``info_blob``.
See below for which keys are accessible for which modifiers.

**Hint:** If you have come up with a new modifier, it might be smart to test if it
actually does what it should with the data.
You can get a ``info_blob`` containing ``x_values`` and ``y_values`` from
the files in your toml list (i.e. before any modifiers have been
applied) like this:

.. code-block:: python

    from orcanet.core import Organizer

    orga = Organizer(output_folder, list_file)
    info_blob = orga.io.get_batch()

This will be exactly what is fed into your modifier when OrcaNet is run, so
testing your new modifier on these will allow you to make sure they work.

Label modifier
--------------
The label modifier is used to generate the labels for the model from the
``y_values`` data of the h5 input files. Unless the label for the model
is directly stored in the h5py files, the definition of a label modifier
is mandatory.

It must be of the following form:

.. code-block:: python

    def my_label_modifier(info_blob):
        ...
        return ys

**Contents of info_blob**:

    ``x_values``, ``y_values``, ``xs``

**Returns**:

    ``ys`` : dict (see above)


It can be set via

.. code-block:: python

    orga.cfg.label_modifier = my_label_modifier

**Hint:** If no label modifier is given, the names of the output layers of
the model have to appear as names of the dtypes in the ``y_values`` recarray.
Then, each output layer will get data from the matching dataset.

Example
^^^^^^^

Assume that we are using this simple classification model with one output,
which is supplied with two different projections of our data at the same time
(XY and ZY):

.. code-block:: python

    inp_1 = Input((1,), name="input_layer_xy")
    inp_2 = Input((1,), name="input_layer_zy")

    x = Concatenate()([inp_1, inp_2])

    output = Dense(2, name="classification")(x)

    example_model = Model((inp_1, inp_2), output)


The output will be either [1,0] or [0,1] (one hot encoding), depending on
whether the event is a neutrino or not.
Suppose that in the mc_info of the input file, one of the
fields has the name ``particle``, which is an int and 1 for neutrinos, or
some other number for non-neutrinos.
We need to convert this to the categorical output of the model with a label
modifier:

.. code-block:: python

    def label_modifier(info_blob):
        y_values = info_blob["y_values"]
        particle = y_values["particle"]
        # Create the label array for the output layer of shape (batchsize, 2)
        ntr_cat = np.zeros(particle.shape + (2, ))
        # If particle is 1, its a neutrino, so we want to have [1,0]
        ntr_cat[:, 0] = particle == 1
        # If particle is not 1, we want [0,1]
        ntr_cat[:, 1] = particle != 1
        # Make a dict to get the label to the correct output layer
        # the output layer is called "classification" in this model
        ys = dict()
        ys["classification"] = ntr_cat
        return ys

Sample modifier
---------------
The sample modifier function is applied to the ``x_values`` dict before it is
fed into the input layers of the network.
It must be of the following form:

.. code-block:: python

    def my_sample_modifier(info_blob):
        ...
        return xs

**Contents of info_blob**:

    ``x_values``, ``y_values``

**Returns**

    ``xs`` : dict (see above)


It can be set via

.. code-block:: python

    orga.cfg.sample_modifier = my_sample_modifier

**Hint:** If no sample modifier is given, the names of the input sets in the
toml list file (= the keys of ``x_values``) and the names of the input layers of
the model have to be
identical. Then, each input layer will get data from the toml input set
with the same name.


Example
^^^^^^^
Using the example classification model from above, assume that we have
input files with data in XY- and in YZ-projections.
In that case, the content of the toml list file could like this::

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

Let's say we want to feed the network XY- and ZY-projections instead, i.e. the
axes of the YZ-projection need to be swapped.
The following sample modifier will perform this operation:

.. code-block:: python

    def sample_modifier(info_blob):
        x_values = info_blob["x_values"]
        xs = dict()

        xs["input_layer_xy"] = x_values["xy"]

        yz_data = x_values["yz"]
        xs["input_layer_zy"] = np.swapaxes(yz_data, 1, 2)  # Axis 0 is the batchsize!

        return xs

Dataset modifier
----------------
The dataset modifiers is only used when a model is evaluated with
``organizer.predict``.
It will determine what is written in the resulting
prediction h5 file.
It must be of the following form:

.. code-block:: python

    def my_dataset_modifier(info_blob)
        ...
        return datasets

**Contents of info_blob**:

    ``y_values``, ``xs``, ``ys``, ``y_pred``


**Returns**

    ``datasets``: ``dict``
        The datasets which will be created in the resulting h5
        prediction file.

        Keys: Names of the datasets.

        Values: The content of each dataset as a numpy array.

It can be set via

.. code-block:: python

    orga.cfg.dataset_modifier = my_dataset_modifier

**Hint:** If no dataset modifier is given, the following datasets will be
created: y_values, and two sets for every output layer (label and pred).
