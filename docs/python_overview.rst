.. _orcanet_python:

OrcaNet python overview
=======================

Using OrcaNet in python happens in two steps:

1. Setting up the organizer with options like batchsize, learning rate, ...
2. Repeated training, validating or predicting with the model.


Step 1: Setting up the Organizer
--------------------------------

The main class of OrcaNet is the Organizer (see
:py:class:`orcanet.core.Organizer`).
It is located in the core module, so it can be set up like this:

.. code-block:: python

    from orcanet.core import Organizer

    organizer = Organizer(output_folder, list_file, config_file)

- ``output_folder`` : str
    The folder where everything will get saved to, i.e. this is where the
    trained models, the log files, the plots etc. will be saved.
    It will be created if it does not exist yet.
- ``list_file`` : str, optional
    Path to a toml file containing a list of the files to be
    trained on. Only necessary for actions requiring a dataset, e.g.
    training, validating or predicting. See :ref:`input_page_orga`
    for the required format of this file.
- ``config_file`` : str, optional
    Optional: Path to a toml file containing new values for the default
    parameters in the configuration member object. See
    :ref:`input_page_orga` for the required format of this file.


All configurable options of the organizer, like the batchsize or the learning
rate, are stored in the Configuration member object
(see :py:class:`orcanet.core.Configuration`).
These options can be changed directly by adressing them, e.g.

.. code-block:: python

    organizer.cfg.batchsize = 32
    organizer.cfg.learning_rate = 0.002
    ...

or by listing them in a toml file::

    [config]
    batchsize=32
    learning_rate=0.002
    ...

and then giving the path to this file as the config_file argument of the
Organizer.

OrcaNet allows for live batchwise modification of samples and labels with
the cfg.sample_modifier and cfg.label_modifier options. See :ref:`modifiers_page`
for details.

Step 2: Working with the model
------------------------------

After the set up, the training can be started via:

.. code-block:: python

    organizer.train_and_validate(model)

This will train the model on all training files in the list_file, while
saving, logging and plotting the progress at the same time.
Then, the model is validated on the validation data, which is also logged
and plotted.

The training and validation could also be executed manually with:

.. code-block:: python

    organizer.train(model)
    organizer.validate()

This will train the given model for one file, and then validate.

To continue a previously started training, run these functions
**without giving a model**. This will make OrcaNet automatically load
the most recent model it can find.

To let the model predict on validation data, use:

.. code-block:: python

    organizer.predict()

This will load the trained and saved model with the lowest validation loss,
and create a h5 file containing for every sample:

- the label for the model
- the prediction of the model
- the mc info block from the val files


Building models with the model builder
--------------------------------------

OrcaNet features a model builder class which can build models from
toml files (see :py:class:`orcanet.model_builder.ModelBuilder`).

It is used as follows:

.. code-block:: python

    from orcanet.model_builder import ModelBuilder

    builder = ModelBuilder(model_file)
    model = builder.build(organizer)

Setting up the model builder is done with ``model_file``, a toml file
containing the info about the model like the number and type of layers.
The format of this file is described on the page
:ref:`input_page_model`.

Building the model requires a set-up organizer, as the input layers of
the model will be adjusted to the data (and possibly present sample
modifiers), so building the model should happen right before the training
or validation is executed.
