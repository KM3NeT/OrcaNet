OrcaNet: A training organizer for Deep Learning in KM3NeT
=========================================================

.. image:: https://git.km3net.de/ml/OrcaNet/badges/master/build.svg
    :target: https://git.km3net.de/ml/OrcaNet/pipelines

.. image:: https://examples.pages.km3net.de/km3badges/docs-latest-brightgreen.svg
    :target: https://ml.pages.km3net.de/OrcaNet

.. image:: https://git.km3net.de/ml/OrcaNet/badges/master/coverage.svg
    :target: https://ml.pages.km3net.de/OrcaNet/coverage


OrcaNet is a deep learning framework based on Keras in order to simplify the 
training process of neural networks for astroparticle physics. It incorporates 
automated logging, plotting and validating during the training, as well as
saving and continuing the training process. Additionally, it features easy 
management of multiple neural network inputs and the use of training data 
which is split over multiple files.

In this sense, it tackles many challenges that are usually found in 
astroparticle physics, like huge datasets.

OrcaNet is a part of the Deep Learning efforts for the neutrino telescope KM3NeT.
Find more information about KM3NeT on http://www.km3net.org

OrcaNet is currently being developed at the official KM3NeT gitlab (https://git.km3net.de/ml/OrcaNet).

However, there's also a github mirror that can be found at https://github.com/ViaFerrata/OrcaNet.

OrcaNet can be installed via pip by running::

    pip install orcanet

By default, orcanet will install tensorflow (the cpu version).
For training with graphics cards, tensorflow-gpu is required, which needs
to be installed manually via::

    pip install tensorflow-gpu

