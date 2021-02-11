"""
Run OrcaNet functionalities from command line.

"""
import argparse

# imports involving tf moved inside functions for speed up


def train(directory, list_file, config_file, model_file=None, to_epoch=None):
    from orcanet.core import Organizer
    from orcanet.model_builder import ModelBuilder

    orga = Organizer(directory, list_file, config_file, tf_log_level=2)

    if orga.io.get_latest_epoch() is None:
        # Start of training
        print("Building new model")
        model = ModelBuilder(model_file).build(orga, verbose=False)
    else:
        model = None

    return orga.train_and_validate(model=model, to_epoch=to_epoch)


def predict(directory, list_file, config_file, epoch=None, fileno=None):
    from orcanet.core import Organizer

    orga = Organizer(directory, list_file, config_file, tf_log_level=1)
    return orga.predict(epoch=epoch, fileno=fileno)[0]


def inference(directory, list_file, config_file, epoch=None, fileno=None):
    from orcanet.core import Organizer

    orga = Organizer(directory, list_file, config_file, tf_log_level=1)
    return orga.inference(epoch=epoch, fileno=fileno)


def main():
    parser = argparse.ArgumentParser(
        prog="orcanet",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers()

    def add_common_args(prsr):
        prsr.add_argument("directory", help="Path to OrcaNet directory.")
        prsr.add_argument("--list_file", help="toml list file")
        prsr.add_argument("--config_file", help="toml config file")

    # orca train
    parser_train = subparsers.add_parser(
        "train",
        description="Train and validate a model.",
    )
    add_common_args(parser_train)
    parser_train.add_argument(
        "--model_file", type=str, help="toml model file", default=None
    )
    parser_train.add_argument(
        "--to_epoch",
        type=int,
        help="Train up to and including this epoch. Default: Train forever.",
        default=None,
    )
    parser_train.set_defaults(func=train)

    # orca pred
    parser_pred = subparsers.add_parser(
        "predict",
        description="Load a trained model and save its prediction on the predictions files to h5.",
    )
    add_common_args(parser_pred)
    parser_pred.add_argument(
        "--epoch", type=int, help="Epoch of model to load. Default: best", default=None
    )
    parser_pred.add_argument(
        "--fileno",
        type=int,
        help="Fileno of model to load. Default: best",
        default=None,
    )
    parser_pred.set_defaults(func=predict)

    # orca inf
    parser_inf = subparsers.add_parser(
        "inference",
        description="Load a trained model and save its prediction on the inference files to h5.",
    )
    add_common_args(parser_inf)
    parser_inf.add_argument(
        "--epoch", type=int, help="Epoch of model to load. Default: best", default=None
    )
    parser_inf.add_argument(
        "--fileno",
        type=int,
        help="Fileno of model to load. Default: best",
        default=None,
    )
    parser_inf.set_defaults(func=inference)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop("func")
    func(**kwargs)
