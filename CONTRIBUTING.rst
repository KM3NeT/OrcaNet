Contributing
============
.. contents:: :local:

Contact Us
----------
Join the Deep Learning channel here: https://chat.km3net.de/channel/deep_learning

Filing Bugs or Feature Requests
-------------------------------

Please **always** create an issue when you encounter any bugs, problems or
need a new feature. Emails and private messages are not meant to communicate
such things!

Use the appropriate template and file a new issue here:
https://git.km3net.de/ml/OrcaNet/issues

If you're not in the KM3NeT collaboration, please open an issue on github:
https://github.com/ViaFerrata/OrcaNet/issues

Please follow the instructions in the templates to provide all the
necessary information which will help other people to understand the
situation.

Make a Fork of OrcaNet
~~~~~~~~~~~~~~~~~~~~~~

You create a fork (your full own copy of the
repository), change the code and when you are happy with the changes, you create
a merge request, so we can review, discuss and add your contribution.
Merge requests are automatically tested on our GitLab CI server and you
don't have to do anything special.

Go to http://git.km3net.de/ml/OrcaNet and click on "Fork".

After that, you will have a full copy of OrcaNet with write access under an URL
like this: ``http://git.km3net.de/your_git_username/OrcaNet``

Clone your Fork to your PC
~~~~~~~~~~~~~~~~~~~~~~~~~~

Get a local copy to work on (use the SSH address `git@git...`, not the HTTP one)::

    git clone git@git.km3net.de:your_git_username/OrcaNet.git

Now you need to add a reference to the original repository, so you can sync your
own fork with the OrcaNet repository::

    cd OrcaNet
    git remote add upstream git@git.km3net.de:ml/OrcaNet.git


Keep your Fork Up to Date
~~~~~~~~~~~~~~~~~~~~~~~~~

To get the most recent commits (including all branches), run::

    git fetch upstream

This will download all the missing commits and branches which are now accessible
using the ``upstream/...`` prefix::

    $ git fetch upstream
    From git.km3net.de:ml/OrcaNet
     * [new branch]        branch1 -> upstream/branch1
     * [new branch]        branch2 -> upstream/branch2


If you want to update for example your **own** ``master`` branch
to contain all the changes on the official ``master`` branch of OrcaNet,
switch to it first with::

    git checkout master

and then merge the ``upstream/master`` into it::

    git merge upstream/master

Make sure to regularly ``git fetch upstream`` and merge changes to your own branches.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DISCLAIMER: This is totally copy & pasted & modified from the excellent km3pipe equivalent.
