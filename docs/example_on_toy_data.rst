Example on toy data
===================

This page shows how to use orcanet via python by applying it on some toy data.
The full scripts can be found in examples/full_example.

In order to use orcanet, data in the form of multi dimensional images in an h5 file,
as well as a compiled keras model is required.

The data
--------

Lets generate some dummy data: Vectors of length 5, filled with random
numbers. The label is the sum of each vector.

.. literalinclude:: ../examples/full_example/full_example.py
   :language: python
   :pyobject: make_dummy_data
   :linenos:


The generated train and val files are given to orcanet with this
toml list file:

.. literalinclude:: ../examples/full_example/example_list.toml
   :language: toml
   :linenos:

Note that we defined the name of this dataset to be "random_numbers".

The model
---------

This is a small compiled keras model with just one hidden dense layer.
Note that the input layer has the same name that we gave the dataset ("random_numbers").

.. literalinclude:: ../examples/full_example/full_example.py
   :language: python
   :pyobject: make_dummy_model
   :linenos:


Training and results
--------------------

After creating the data and compiling the model, they are handed to the
Organizer object. Training and validation, as well as predicting can be
done in just one line each.

In total, the generation of the model, the data, and conducting the
training is done like this:

.. literalinclude:: ../examples/full_example/full_example.py
   :language: python
   :pyobject: use_orcanet
   :linenos:
