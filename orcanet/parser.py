"""
Run OrcaNet functionalities from command line.

"""
import argparse
# imports involving tf moved inside functions for speed up


def train(directory, list_file=None, config_file=None, model_file=None, to_epoch=None):
    from orcanet.core import Organizer
    from orcanet.model_builder import ModelBuilder
    from orcanet.misc import find_file

    orga = Organizer(directory, list_file, config_file, tf_log_level=1)

    if orga.io.get_latest_epoch() is None:
        # Start of training
        print("Building new model")
        if model_file is None:
            model_file = find_file(directory, "model.toml")
        model = ModelBuilder(model_file).build(orga, verbose=False)
    else:
        model = None

    return orga.train_and_validate(model=model, to_epoch=to_epoch)


def _add_parser_train(subparsers):
    parser = subparsers.add_parser(
        "train",
        description="Train and validate a model.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--model_file",
        type=str,
        help="Path to toml model file. Will be used to build a model at "
             "the start of the training. Not needed to resume training. "
             "Default: Look for a file called 'model.toml' in the "
             "given OrcaNet directory.",
        default=None,
    )
    parser.add_argument(
        "--to_epoch",
        type=int,
        help="Train up to and including this epoch. Default: Train forever.",
        default=None,
    )
    parser.set_defaults(func=train)


def predict(directory, list_file=None, config_file=None, epoch=None, fileno=None):
    from orcanet.core import Organizer

    orga = Organizer(directory, list_file, config_file, tf_log_level=1)
    return orga.predict(epoch=epoch, fileno=fileno)[0]


def _add_paser_predict(subparsers):
    parser = subparsers.add_parser(
        "predict",
        description="Load a trained model and save its prediction on "
                    "the predictions files to h5.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch of model to load. Default: best",
        default=None
    )
    parser.add_argument(
        "--fileno",
        type=int,
        help="Fileno of model to load. Default: best",
        default=None,
    )
    parser.set_defaults(func=predict)


def inference(directory, list_file=None, config_file=None, epoch=None, fileno=None):
    from orcanet.core import Organizer

    orga = Organizer(directory, list_file, config_file, tf_log_level=1)
    return orga.inference(epoch=epoch, fileno=fileno)


def _add_parser_inference(subparsers):
    parser = subparsers.add_parser(
        "inference",
        description="Load a trained model and save its prediction on the "
                    "inference files to h5.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch of model to load. Default: best",
        default=None,
    )
    parser.add_argument(
        "--fileno",
        type=int,
        help="Fileno of model to load. Default: best",
        default=None,
    )
    parser.set_defaults(func=inference)


def _add_common_args(prsr):
    prsr.add_argument(
        "directory",
        help="Path to OrcaNet directory.",
    )
    prsr.add_argument(
        "--list_file",
        type=str,
        help="Path to toml list file. Default: Look for a file called "
             "'list.toml' in the given OrcaNet directory.",
        default=None,
    )
    prsr.add_argument(
        "--config_file",
        type=str,
        help="Path to toml config file. Default: Look for a file called "
             "'config.toml' in the given OrcaNet directory.",
        default=None,
    )


def _add_parser_summarize(subparsers):
    import orcanet.utilities.summarize_training as summarize_training

    parent_parser = summarize_training.get_parser()
    parser = subparsers.add_parser(
        "summarize",
        description=parent_parser.description,
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[parent_parser],
        add_help=False,
    )
    parser.set_defaults(func=summarize_training.summarize)


def _add_parser_version(subparsers):
    def show_version():
        from orcanet import version
        print(version)

    parser = subparsers.add_parser(
        "version",
        description="Show installed orcanet version.",
    )
    parser.set_defaults(func=show_version)


def main():
    parser = argparse.ArgumentParser(
        prog="orcanet",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers()

    _add_parser_train(subparsers)
    _add_paser_predict(subparsers)
    _add_parser_inference(subparsers)
    _add_parser_summarize(subparsers)
    _add_parser_version(subparsers)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop("func")
    func(**kwargs)
