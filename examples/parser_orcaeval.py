"""
Evaluate a model with a parser.

Usage:
    parser_orcaeval.py FOLDER LIST CONFIG MODEL
    parser_orcaeval.py (-h | --help)

Arguments:
    FOLDER  Path to the folder where everything gets saved to, e.g. the summary.txt, the plots, the trained models, etc.
    LIST    A .toml file which contains the pathes of the training and validation files.
            An example can be found in examples/settings_files/example_list.toml
    CONFIG  A .toml file which sets up the training.
            An example can be found in examples/settings_files/example_config.toml. The possible parameters are listed in
            core.py in the class Configuration.
    MODEL   Path to a .toml file with infos about a model.
            An example can be found in examples/settings_files/example_model.toml.

Options:
    -h --help                       Show this screen.

"""
from docopt import docopt
from orcanet.core import Configuration
from orcanet.utilities.losses import get_all_loss_functions
from orcanet_contrib.contrib import orca_label_modifiers, orca_sample_modifiers


def orca_eval(main_folder, list_file, config_file):
    """
    This shows how to use OrcaNet.

    Parameters
    ----------
    main_folder : str
        Path to the folder where everything gets saved to, e.g. the summary log file, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and validation.
    config_file : str
        Path to a .toml file which overwrites some of the default settings for training and validating a model.

    """
    # Set up the cfg object with the input data
    cfg = Configuration(main_folder, list_file, config_file)
    # Orca networks use some custom loss functions, which need to be handed to keras when loading models
    cfg.custom_objects = get_all_loss_functions()
    # Set the sample and label modifiers needed for feeding data into the network
    model_data = cfg.get_modeldata()
    if model_data.swap_4d_channels is not None:
        cfg.sample_modifier = orca_sample_modifiers(model_data.swap_4d_channels, model_data.str_ident)
    cfg.label_modifier = orca_label_modifiers(model_data.class_type)
    # Per default, an evaluation will be done for the model with the highest epoch and filenumber.
    # Can be adjusted with cfg.eval_epoch and cfg.eval_fileno
    cfg.eval(cfg)


def parse_input():
    """ Run the cfg.eval function with a parser. """
    args = docopt(__doc__)
    main_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    orca_eval(main_folder, list_file, config_file)


if __name__ == '__main__':
    parse_input()
