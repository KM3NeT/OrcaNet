"""
Use orca_train with a parser.

Usage:
    parser_orcatrain.py [options] FOLDER LIST CONFIG MODEL
    parser_orcatrain.py (-h | --help)

Arguments:
    FOLDER  Path to the folder where everything gets saved to, e.g. the
            summary.txt, the plots, the trained models, etc.
    LIST    A .toml file which contains the pathes of the training and
            validation files. An example can be found in
            examples/example_list.toml
    CONFIG  A .toml file which sets up the training. An example can be
            found in examples/example_config.toml. The possible parameters
            are listed in core.py in the class Configuration.
    MODEL   Path to a .toml file with infos about a model.
            An example can be found in examples/example_model.toml.

Options:
    -h --help    Show this screen.
    --recompile  Recompile the keras model, e.g. needed if the loss weights
                 are changed during the training.

"""
from docopt import docopt

from orcanet.core import OrcaHandler
from orca_builder import OrcaBuilder
from orcanet_contrib.orca_handler_util import orca_learning_rates, update_orca_objects


def orca_train(output_folder, list_file, config_file, model_file, recompile_model=False):
    """
    Run orca.train with predefined OrcaBuilder networks using a parser.

    Parameters
    ----------
    output_folder : str
        Path to the folder where everything gets saved to, e.g. the summary
        log file, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that
        should be used for training and validation.
    config_file : str
        Path to a .toml file which overwrites some of the default settings
        for training and validating a model.
    model_file : str
        Path to a file with parameters to build a model of a predefined
        architecture with OrcaNet.
    recompile_model : bool
        If the model should be recompiled or not. Necessary, if e.g. the
        loss_weights are changed during the training.

    """
    # Set up the OrcaHandler with the input data
    orca = OrcaHandler(output_folder, list_file, config_file)

    # Load in the orca sample-, label-, and dataset-modifiers, as well as
    # the custom objects
    update_orca_objects(orca, model_file)

    # If this is the start of the training, a compiled model needs to be
    # handed to the orca_train function
    if orca.io.is_new():
        # The OrcaBuilder class allows to construct models from a toml file,
        # adapted to the datasets in the orca instance. Its modifiers will
        # be taken into account for this
        builder = OrcaBuilder(model_file)
        model = builder.build(orca)

    else:
        model = None

        if recompile_model is True:
            builder = OrcaBuilder(model_file)
            model = builder.recompile_model(orca)

    # Use a custom LR schedule
    orca.cfg.learning_rate = orca_learning_rates("triple_decay",
                                                 orca.io.get_no_of_files("train"))

    # start the training
    orca.train(model=model, force_model=recompile_model)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    output_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    model_file = args['MODEL']
    recompile_model = args['--recompile']
    orca_train(output_folder, list_file, config_file, model_file, recompile_model=recompile_model)


if __name__ == '__main__':
    parse_input()
