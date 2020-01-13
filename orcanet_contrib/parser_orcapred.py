"""
Use orga.predict with a parser.

Usage:
    parser_orcapred.py FOLDER LIST CONFIG MODEL [--epoch EPOCH] [--fileno FILENO]
    parser_orcapred.py (-h | --help)

Arguments:
    FOLDER  Path to the folder where everything gets saved to, e.g. the
            summary.txt, the plots, the trained models, etc.
    LIST    A .toml file which contains the pathes of the training and
            validation files. An example can be found in
            examples/list_file.toml
    CONFIG  A .toml file which sets up the training. An example can be
            found in examples/config_file.toml. The possible parameters
            are listed in core.py in the class Configuration.
    MODEL   Path to a .toml file with infos about a model.
            An example can be found in examples/explanation.toml.

Options:
    -h --help        Show this screen.
    --epoch=EPOCH    Use model of given epoch. [default: None]
    --fileno=FILENO  Use model of given fileno. [default: None]

"""
from matplotlib import use
use('Agg')

from docopt import docopt

from orcanet.core import Organizer
from orcanet_contrib.orca_handler_util import update_objects


def orca_pred(output_folder, list_file, config_file, model_file,
              epoch=None, fileno=None):
    """
    Run orga.predict with predefined ModelBuilder networks using a parser.

    Per default, the most recent saved model will be loaded.

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
    epoch : int, optional
        The epoch of the saved model to predict with.
    fileno : int, optional
        The filenumber of the saved model to predict with.

    """
    # Set up the Organizer with the input data
    orga = Organizer(output_folder, list_file, config_file, tf_log_level=1)

    # When predicting with a orga model, the right modifiers and custom
    # objects need to be given
    update_objects(orga, model_file)

    # Per default, a prediction will be done for the model with the
    # highest epoch and filenumber.
    orga.predict(epoch=epoch, fileno=fileno, concatenate=True)


def main():
    """ Run the orca_pred function with a parser. """
    args = docopt(__doc__)
    if args["--epoch"] == "None":
        epoch = None
    else:
        epoch = int(args["--epoch"])

    if args["--fileno"] == "None":
        fileno = None
    else:
        fileno = int(args["--fileno"])

    orca_pred(output_folder=args['FOLDER'],
              list_file=args['LIST'],
              config_file=args['CONFIG'],
              model_file=args['MODEL'],
              epoch=epoch,
              fileno=fileno)


if __name__ == '__main__':
    main()
