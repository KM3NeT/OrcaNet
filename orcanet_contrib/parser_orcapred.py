"""
Use orca_pred with a parser.

Usage:
    parser_orcapred.py FOLDER LIST CONFIG MODEL
    parser_orcapred.py (-h | --help)

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

from orcanet.core import OrcaHandler
from orcanet.model_archs.model_setup import OrcaModel
from orcanet_contrib.eval_nn import make_performance_plots
from orcanet_contrib.contrib import orca_dataset_modifiers


def orca_pred(output_folder, list_file, config_file, model_file):
    """
    Run orca.predict with predefined OrcaModel networks using a parser.

    Parameters
    ----------
    output_folder : str
        Path to the folder where everything gets saved to, e.g. the summary log file, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and validation.
    config_file : str
        Path to a .toml file which overwrites some of the default settings for training and validating a model.
    model_file : str
        Path to a file with parameters to build a model of a predefined architecture with OrcaNet.

    """
    # Set up the OrcaHandler with the input data
    orca = OrcaHandler(output_folder, list_file, config_file)

    # When predicting with a orca model, the right modifiers and custom objects need to be given
    orcamodel = OrcaModel(model_file)
    orcamodel.update_orca(orca)

    # get dataset modifers for predicting
    dataset_modifier = orca_dataset_modifiers(orcamodel.class_type)
    orca.cfg.dataset_modifier = dataset_modifier

    # Per default, an evaluation will be done for the model with the highest epoch and filenumber.
    # Can be adjusted with cfg.eval_epoch and cfg.eval_fileno
    pred_filename = orca.predict()

    plots_folder = orca.io.get_subfolder(name='plots')
    make_performance_plots(pred_filename, orcamodel.class_type, plots_folder)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    output_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    model_file = args['MODEL']
    orca_pred(output_folder, list_file, config_file, model_file)


if __name__ == '__main__':
    parse_input()
