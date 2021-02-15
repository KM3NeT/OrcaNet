Quick start
===========
Learn how to use OrcaNet in just 4 simple steps!
This assumes you have already installed orcanet.

Step 1
------

Create a directory for your training with an adequate name.
E.g.::

    mkdir my_first_orcanet_training

This directory will contain all the logfiles, plots, checkpoints etc. of
this training.


Step 2
------

Create your toml configuration files and move them into your directory.
These toml files describe:

- The data that will be used (``list.toml``)
- The configuration of the training (``config.toml``)
- The architecture of the model (``model.toml``)

See :ref:`input_page` for more details on these files.
Make sure to give them these exact names
(``list.toml``, ``config.toml``, ``model.toml``).
This allows OrcaNet to automatically discover them in your directory.


Step 3
------

Start the training like this::

    orcanet train my_first_orcanet_training

You can monitor how the training goes by looking at the summary plot in
my_first_orcanet_training/plots/summary_plot.pdf.
Or, if you prefer an interactive plot, with this command::

    orcanet summarize my_first_orcanet_training

To resume a training, just run the train command again.
Once the model has fully converged, you can continue with the next step.


Step 4
------

Save the output of the model to hdf5 like this::

    orcanet predict my_first_orcanet_training

or like this::

    orcanet inference my_first_orcanet_training

Depending on wether you want to use the validation files or the
inference files as an input.
The result gets saved to ``my_first_orcanet_training/predictions``.


Further Info
------------
If you want to learn how to use the python interface of OrcaNet, you can
check out :ref:`orcanet_python`.
