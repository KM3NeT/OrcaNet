Modifiers in OrcaNet
====================
.. contents:: :local:

Modifiers are used to preprocess data from the input files, before handing them
to the network.
They are functions which are applied to both the samples, as well as the labels.
They require the input and output layers of the network, as well as ever input
set in the toml list file to be named.
This makes it easy to assure that the right data is fed into the right layer of
the network, especially if there are multiple inputs or outputs.

**Hint:** If you have developed a new modifier, it might be smart to test if it
actually does what it should with the data.
You can get a batch of samples ``xs`` and ``mc_info`` data from your toml
list file like this:

.. code-block:: python

    from orcanet.core import Organizer

    orga = Organizer(output_folder, list_file)
    xs, mc_info = orga.io.get_batch()

The modifiers can then be applied to them.

Label modifier
--------------
The label modifier is used to generate the labels for the model from the
``mc_info`` data of the h5 input files. Unless the label for the model
is directly stored in the h5py files, the definition of a label modifier
is mandatory.
It must be of the following form:

.. code-block:: python

    def label_modifier(mc_info):
        ...
        return y_true

``mc_info``: ``numpy array``
    One batch of data read from the ``key_mc_info`` datagroup in the h5 input
    files.
    If the content of the datagroup is a structured array, this will
    also be a structured array.
``y_true``: ``dict``
    Keys: The names of the output layers of the model.

    Values: One batch of labels as a numpy array.

It can be set via

.. code-block:: python

    orga.cfg.label_modifier = label_modifier

**Hint:** If the names of the output layers of the model are also the
names of dtypes in the ``mc_info`` data, no label modifier is required!

Example
^^^^^^^

A simple classification model with one output.

.. code-block:: python

    def get_example_model():
        inp_1 = Input((1,), name="input_layer_xy")
        inp_2 = Input((1,), name="input_layer_zy")

        x = Concatenate()([inp_1, inp_2])

        output = Dense(2, name="classification")(x)

        example_model = Model((inp_1, inp_2), output)
        return example_model

The output will be either [1,0] or [0,1] (one hot encoding), depending on
whether the event is a neutrino or not.
Suppose that in the structured array mc_info of the input file, one of the
fields has the name ``particle``, which is an int and 1 for neutrinos, or
some other number for non-neutrinos.
We need to convert this to the categorical output of the model with a label
modifier:

.. code-block:: python

    def label_modifier(mc_info):
        particle = mc_info["particle"]
        # Create the label array for the output layer of shape (batchsize, 2)
        ntr_cat = np.zeros(particle.shape + (2, ))
        # [1,0] for neutrinos, [0,1] for not neutrinos
        ntr_cat[:, 0] = particle == 1
        ntr_cat[:, 1] = particle != 1
        # Make a dict to get the label to the correct output layer
        y_true = dict()
        y_true["classification"] = ntr_cat
        return y_true

Sample modifier
---------------
The sample modifier is used to lead the samples read from the h5 input
file to the input layers of the network.
It must be of the following form:

.. code-block:: python

    def sample_modifier(xs_files):
        ...
        return xs_layer

``xs_files``: ``dict``
    One batch of data read from the ``key_samples`` datagroup in the h5 input
    files.

    Keys: Input set names from the toml list file.

    Values: Numpy array with samples from the respective file.
``xs_layer``: ``dict``
    One batch of data on which the model will be trained on.

    Keys: Name of an input layer of the network.

    Values: Numpy array with samples.

It can be set via

.. code-block:: python

    orga.cfg.sample_modifier = sample_modifier

**Hint:** If the names of the input sets in the toml list file and the names of
the input layers match, no sample modifier is required!


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

    def sample_modifier(xs_list):
        xs_layer = dict()
        xs_layer["input_layer_xy"] = xs_list["xy"]
        yz_data = xs_list["yz"]
        xs_layer["input_layer_zy"] = np.swapaxes(yz_data, 1, 2)  # Axis 0 is the batchsize!
        return xs_layer

Dataset modifier
----------------
The dataset modifiers is only used when a model is evaluated with
``organizer.predict``.
It will determine what is written in the resulting
prediction h5 file.
It must be of the following form:

.. code-block:: python

    def dataset_modifier(mc_info, y_true, y_pred)
        ...
        return datasets

``mc_info``: ``numpy structured array``
    One batch of data read from the ``key_mc_info`` datagroup in the h5 input
    files.
    If the content of the datagroup is a structured array, this will
    also be a structured array.
``y_true``: ``dict``
    The labels given to the model.

    Keys: The names of the output layers of the model.

    Values: One batch of labels as a numpy array.
``y_pred``: ``dict``
    The predictions of the model.

    Keys: The names of the output layers of the model.

    Values: One batch of predictions from the respective output layer of the
    model as a numpy array.
``datasets``: ``dict``
    The datasets which will be created in the resulting h5
    prediction file.

    Keys: Names of the datasets.

    Values: The content of each dataset as a numpy array.

It can be set via

.. code-block:: python

    orga.cfg.dataset_modifier = dataset_modifier

**Hint:** If no dataset modifier is given, the following datasets will be
created: mc_info, and two sets for every output layer (label and pred).
