.. _input_page:

Input toml files
================

Much of the input in OrcaNet is given as toml files.
This page shows examples for the proper format, as well as possible options.

.. _input_page_orga:

For the Organizer
-----------------

The :py:class:`orcanet.core.Organizer` takes a ``list_file`` and a
``config_file`` argument, which are the pathes to the respective files.

Here are examples for the proper file formats:

list_file
^^^^^^^^^

.. literalinclude:: ../examples/list_file.toml
   :language: toml
   :linenos:
   :caption: examples/list_file.toml


config_file
^^^^^^^^^^^

.. literalinclude:: ../examples/config_file.toml
   :language: toml
   :linenos:
   :caption: examples/config_file.toml

.. _input_page_model:

For the model builder
---------------------

Models can be build using toml input files. The path is handed over as
the ``model_file`` argument of
:py:class:`orcanet.model_builder.ModelBuilder`.

.. code-block:: python

    from orcanet.model_builder import ModelBuilder

    mb = ModelBuilder(model_file="model.toml")
    model = mb.build(organizer)

Here is an example for the proper file format
(See :ref:`example_models_page` for more examples):

.. literalinclude:: ../examples/model_files/explanation.toml
   :language: toml
   :linenos:
   :caption: examples/model_files/explanation.toml

