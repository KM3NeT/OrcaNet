Unreleased changes
------------------

Version 0
---------

0.7.1 / 2019-07-17
~~~~~~~~~~~~~~~~~~
* fix for error when using pred after inf

0.7 / 2019-07-15
~~~~~~~~~~~~~~~~
BREAKING CHANGES:
* All modifiers take a dict (info_blob) as an input now, which has the old values as keys, e.g. xs is now info_blob["xs"]
* mc_info for the modifiers is now called y_values
* key_mc_info is now called key_y_values
* key_samples is now called key_x_values

Other changes:
* Rework generator
* Added Resnet Blocks
* Added summarize function + entry point to quickly and interactivly plot and compare training progresses.
* Added cleanup method and option to automatically save a lot of disk space
* Added Spatial dropout layer
* Model summary is printed to log.txt
* Faster summary plot plotting
* Rework of train/val plotter
* Expanded doc
* Add option to log the compile options in the ModelBuilder class, when a model is compiled.
* Add support for xyz-t + xyz-p (vertically stacked) and yzt-x double input, pre-trained model
* New plots in orcanet_contrib for regression & ts and minor other stuff.


0.6 / 2019-05-27
~~~~~~~~~~~~~~~~
* Changed plot_history default line width and marker size.
* Bugfix for swap_columns sample_modifier in orca_handler_util
* Fixed filter_tf_garbage option. The tf CPP_MIN_LOG_LEVEl variable should now be set when instantiating the Organizer.
* Added epoch/fileno to orcapred.
* Added possibility to define inference files.
* Pred and inference will now automatically look for the best model to load.
* Various fixes

0.5 / 2019-05-14
~~~~~~~~~~~~~~~~~~~
* Various fixes and small additions
* model are automatically dumped as json
* support for csv learning rate files

0.4.1 / 2019-04-17
~~~~~~~~~~~~~~~~~~~
* Release of v0.4.1
