OrcaNet: A training organizer for Deep Learning in KM3NeT
=========================================================

.. image:: https://badge.fury.io/py/orcanet.svg
    :target: https://badge.fury.io/py/orcanet

.. image:: https://git.km3net.de/ml/OrcaNet/badges/master/pipeline.svg
    :target: https://git.km3net.de/ml/OrcaNet/pipelines

.. image:: https://examples.pages.km3net.de/km3badges/docs-latest-brightgreen.svg
    :target: https://ml.pages.km3net.de/OrcaNet

.. image:: https://git.km3net.de/ml/OrcaNet/badges/master/coverage.svg
    :target: https://ml.pages.km3net.de/OrcaNet/coverage

.. image:: https://api.codacy.com/project/badge/Grade/6c81a8396eb34a9d88f07b6620535432
    :alt: Codacy Badge
    :target: https://www.codacy.com/app/sreck/OrcaNet?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=StefReck/OrcaNet&amp;utm_campaign=Badge_Grade


OrcaNet is a deep learning framework based on tensorflow in order to simplify the
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


In order to make use of tensorflow's GPU acceleration, you need cuda and
cudnn installed. You can see which of these each tensorflow version needs here
https://www.tensorflow.org/install/source#gpu


Containerization
----------------
A Singularity image of the latest stable version of OrcaNet with tf and cuda/cudnn
is automatically uploaded to our sftp server.
Download it via
``wget http://pi1139.physik.uni-erlangen.de/singularity/orcanet.sif``
and run it e.g. via
``singularity shell orcanet.sif``


Tensorflow 1.X
--------------

Orcanet per default runs on tensorflow 2, but there is also a no longer supperted
tf 1 branch `tf1`, which makes use of the deprecated stand-alone keras package.
For the tf 1 version, orcanet will install tensorflow (the cpu version).
For training with graphics cards, tensorflow-gpu is required, which needs
to be installed manually via::

    pip install tensorflow-gpu

