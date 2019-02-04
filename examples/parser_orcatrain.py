"""
Use orca_train with a parser.

Usage:
    parser_orcatrain.py FOLDER LIST CONFIG MODEL
    parser_orcatrain.py (-h | --help)

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
from orcanet.core import orca_train, Configuration
from orcanet.model_archs.model_setup import build_nn_model
from orcanet.utilities.losses import get_all_loss_functions


def run_train(main_folder, list_file, config_file, model_file):
    """
    Use orca_train with a parser.

    Parameters
    ----------
    main_folder : str
        Path to the folder where everything gets saved to, e.g. the summary log file, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and validation.
    config_file : str
        Path to a .toml file which overwrites some of the default settings for training and validating a model.
    model_file : str
        Path to a file with parameters to build a model of a predefined architecture with OrcaNet.

    """
    # Set up the cfg object with the input data
    cfg = Configuration(main_folder, list_file, config_file)
    # Orca networks use some custom loss functions, which need to be handed to keras when loading models
    cfg.custom_objects = get_all_loss_functions()
    # If this is the start of the training, a compiled model needs to be handed to the orca_train function
    if cfg.get_latest_epoch() == (0, 0):
        # Add Info for building a model with OrcaNet to the cfg object
        cfg.set_from_model_file(model_file)
        # Build it
        initial_model = build_nn_model(cfg)
    else:
        # No model is required if the training is continued, as it will be loaded automatically
        initial_model = None
    orca_train(cfg, initial_model)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    main_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    model_file = args['MODEL']
    run_train(main_folder, list_file, config_file, model_file)


if __name__ == '__main__':
    parse_input()
