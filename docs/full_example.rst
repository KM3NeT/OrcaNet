Working example
===============

This page shows how to use orcanet by applying it on dummy data.
The full scripts can be found in examples/full_example.

The data
--------

Lets generate some dummy data: Vectors of length 5, filled with random
numbers. The label is the sum of each vector.

.. literalinclude:: ../examples/full_example/full_example.py
   :language: python
   :pyobject: make_dummy_data
   :linenos:


The generated train and val files are given to the model with this
list file:

.. literalinclude:: ../examples/full_example/example_list.toml
   :language: toml
   :linenos:


The model
---------

This is a small model with just one hidden dense layer.

.. literalinclude:: ../examples/full_example/full_example.py
   :language: python
   :pyobject: make_dummy_model
   :linenos:


Training and results
--------------------

In total, the generation of the model, the data, and conducting the
training is done like this:

.. literalinclude:: ../examples/full_example/full_example.py
   :language: python
   :pyobject: use_orcanet
   :linenos:
