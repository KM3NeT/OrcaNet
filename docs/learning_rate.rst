Learning rate
=============

OrcaNet provides multiple ways to easily determine a learning rate schedule
for the training. For this, a
learning rate varibale is stored in ``organizer.cfg.learning_rate``, which is set
to None per default (no change to the current learning rate of the model).
Depending on what type this attribute is, the learning rate schedule
during the training will be one of the following:

Float
*****
The learning rate will be constantly this value, for all epochs.

Tuple
*****
A tuple (or list) of two floats: The first float gives the learning rate
in epoch 1 file 1, and the second float gives the decrease of the
learning rate per file.

For example, if ``organizer.cfg.learning_rate`` = [0.1, 0.4] is used,
the learning rates will be 0.1, 0.06, 0.036, ...

Function
********
A custom learning rate schedule.
The function has to have exactly two input parameters:
The epoch and the file number (in this order).
It must return the learning rate for this (epoch, fileno) pair.

String
******
A custom learning rate schedule in the form of a txt document.
It is the path to a csv file inside the output folder the organizer was initialized with.
This file must contain 3 columns with the epoch, fileno, and the value the lr
will be set
to when reaching this epoch/fileno.

An example can be found in orcanet/examples/learning_rate.csv:

.. literalinclude:: ../examples/learning_rate.csv
   :linenos:

