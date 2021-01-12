"""
Use orga.train with a parser.

Usage:
    parser_orcatrain.py [options] FOLDER LIST CONFIG MODEL EPOCHS
    parser_orcatrain.py (-h | --help)

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
    EPOCHS  The number of epochs to train.

Options:
    -h --help    Show this screen.
    --recompile  Recompile the keras model, e.g. needed if the loss weights
                 are changed during the training.

"""
from matplotlib import use
use('Agg')

import warnings
import numpy as np
from docopt import docopt
import tensorflow.keras as ks
import toml

from orcanet.core import Organizer
from orcanet.model_builder import ModelBuilder
from orcanet_contrib.orca_handler_util import orca_learning_rates, update_objects, GraphSampleMod, GraphSampleMod_only_first_hit


def orca_train(output_folder, list_file, config_file, model_file,no_epochs,
               recompile_model=False):
    """
    Run orga.train with predefined ModelBuilder networks using a parser.

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
    # Set up the Organizer with the input data
    orga = Organizer(output_folder, list_file, config_file, tf_log_level=1)
    
    #special sample modifier for the graph neural networks; dont use the "sample_modifier" 
    #option in the model.toml file
    file_content = toml.load(model_file)
    knn_for_sample_mod = file_content["model"]["next_neighbors"]
    
    #load a specific sample mod, even for gnn
    sample_modifier = file_content["orca_modifiers"]["sample_modifier_own"]
    
    if sample_modifier == "normal":
        orga.cfg.sample_modifier = GraphSampleMod(knn=knn_for_sample_mod)
    elif sample_modifier == "only_first_hit":
        orga.cfg.sample_modifier = GraphSampleMod_only_first_hit(knn=knn_for_sample_mod)
    else:
        "No valid sample_modifier_own given!"
        exit()
        
    # Load in the orga sample-, label-, and dataset-modifiers, as well as
    # the custom objects
    update_objects(orga, model_file)

    # If this is the start of the training, a compiled model needs to be
    # handed to the orga.train function
    if orga.io.get_latest_epoch() is None:
        # The ModelBuilder class allows to construct models from a toml file,
        # adapted to the datasets in the orga instance. Its modifiers will
        # be taken into account for this
        builder = ModelBuilder(model_file)
        model = builder.build(orga, log_comp_opts=True)

    elif recompile_model is True:
        builder = ModelBuilder(model_file)

        path_of_model = orga.io.get_model_path(-1, -1)
        model = ks.models.load_model(path_of_model,
                                     custom_objects=orga.cfg.get_custom_objects())
        print("Recompiling the saved model")
        model = builder.compile_model(model, custom_objects=orga.cfg.get_custom_objects())
        builder.log_model_properties(orga)

    else:
        model = None

    try:
        # Use a custom LR schedule
        user_lr = orga.cfg.learning_rate
        lr = orca_learning_rates(user_lr, orga.io.get_no_of_files("train"))
        orga.cfg.learning_rate = lr
    except NameError:
        pass
    

    
    # start the training
    orga.train_and_validate(model=model,epochs=int(no_epochs))


def main():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    output_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    model_file = args['MODEL']
    no_epochs = args['EPOCHS']
    recompile_model = args['--recompile']
    orca_train(output_folder, list_file, config_file, model_file,no_epochs,
               recompile_model=recompile_model)


if __name__ == '__main__':
    main()
