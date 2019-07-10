from unittest import TestCase
from orcanet.core import Organizer
import orcanet.model_builder
from orcanet_contrib.custom_objects import get_custom_objects
import os


class TestSequentialBuilder(TestCase):
    """
    Just build some models to see if there are embarrassing formatting errors.
    """
    def setUp(self):
        orca_dir = os.path.dirname(os.path.dirname(orcanet.model_builder.__file__))
        self.example_dir = os.path.join(orca_dir, "examples", "model_files")

        def get_orga(dims):
            orga = Organizer(".")
            if dims == 2:
                orga.io.get_input_shapes = get_input_shapes_2d
            elif dims == 3:
                orga.io.get_input_shapes = get_input_shapes_3d
            else:
                raise AssertionError
            return orga
        self.get_orga = get_orga

    def test_cnn(self):
        toml_file = "cnn.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        model = mb.build(self.get_orga(dims=2))

    def test_resnet(self):
        toml_file = "resnet.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        model = mb.build(self.get_orga(dims=2))

    def test_explanation(self):
        toml_file = "explanation.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        orga = self.get_orga(dims=3)
        orga.cfg.custom_objects = get_custom_objects()
        model = mb.build(orga)


def get_input_shapes_3d():
    dims = (10, 10, 10, 1)
    return {"input_A": dims}


def get_input_shapes_2d():
    dims = (10, 10, 1)
    return {"input_A": dims}
