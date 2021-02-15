.. _input_page:

Toml files
==========

Much of the input in OrcaNet is given as toml files.
This page shows examples for the proper format, as well as possible options.

.. _input_page_orga:

toml files for the Organizer
----------------------------

The :py:class:`orcanet.core.Organizer` takes a ``list_file`` and a
``config_file`` argument, which are the pathes to the respective files.

Here are examples for the proper file formats:

list_file
^^^^^^^^^

.. literalinclude:: ../examples/list.toml
   :language: toml
   :linenos:
   :caption: examples/list.toml


config_file
^^^^^^^^^^^

An important paramter of the config files are the modifiers. Check out
:ref:`modifiers_page` for more info. You can find various built-in modifiers
in :py:mod:`orcanet.lib`.

.. literalinclude:: ../examples/config.toml
   :language: toml
   :linenos:
   :caption: examples/config.toml

.. _input_page_model:

toml files for the model builder
--------------------------------

Models can be built with OrcaNet using a toml file.
The path is handed over as the ``model_file`` argument of
:py:class:`orcanet.model_builder.ModelBuilder`.
You can find various built-in layer blocks
in :py:mod:`orcanet.builder_util.layer_blocks`,
and some built-in losses in :py:mod:`orcanet.lib.losses`.

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

