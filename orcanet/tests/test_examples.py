from unittest import TestCase
from orcanet.core import Organizer
import orcanet.model_builder
import os


class TestSequentialBuilder(TestCase):
    """
    Just build some models to see if there are embarrassing formatting errors.
    """
    def setUp(self):
        orca_dir = os.path.dirname(os.path.dirname(orcanet.model_builder.__file__))
        self.example_dir = os.path.join(orca_dir, "examples", "model_files")

        def get_orga(dims):
            dimdict = {
                2: get_input_shapes_2d,
                3: get_input_shapes_3d,
                "graph": get_input_shapes_graph}
            orga = Organizer(".")
            orga.io.get_input_shapes = dimdict[dims]
            return orga
        self.get_orga = get_orga

    def test_cnn(self):
        toml_file = "cnn.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        model = mb.build(self.get_orga(dims=2))

        self.assertSequenceEqual(model.output_shape, (None, 3))
        self.assertEqual(model.count_params(), 4691523)

    def test_resnet(self):
        toml_file = "resnet.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        model = mb.build(self.get_orga(dims=2))

        self.assertSequenceEqual(model.output_shape, (None, 3))
        self.assertEqual(model.count_params(), 11141699)

    def test_explanation(self):
        toml_file = "explanation.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        orga = self.get_orga(dims=3)
        model = mb.build(orga)

        self.assertSequenceEqual(model.output_shape, (None, 3))
        self.assertEqual(model.count_params(), 2109635)

    def test_inception(self):
        toml_file = "inception.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        model = mb.build(self.get_orga(dims=2))

        self.assertSequenceEqual(model.output_shape, (None, 3))
        self.assertEqual(model.count_params(), 149827)

    def test_lstm(self):
        toml_file = "lstm.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        model = mb.build(self.get_orga(dims=3))

        self.assertSequenceEqual(model.output_shape, (None, 3))
        self.assertEqual(model.count_params(), 11321)

    def test_medgeconv(self):
        toml_file = "graph_medgeconv.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        orga = self.get_orga(dims="graph")
        orga.cfg.batchsize = 64
        orga.cfg.fixed_batchsize = True
        model = mb.build(orga)

        self.assertSequenceEqual(model.output_shape, (64, 3))
        self.assertEqual(model.count_params(), 304223)
        self.assertEqual(len(model.layers), 87)

    def test_disjoint_edgeconv(self):
        toml_file = "graph_disjoint_edgeconv.toml"

        model_file = os.path.join(self.example_dir, toml_file)
        mb = orcanet.model_builder.ModelBuilder(model_file)
        orga = self.get_orga(dims="graph")
        orga.cfg.batchsize = 64
        orga.cfg.fixed_batchsize = True
        model = mb.build(orga)

        self.assertSequenceEqual(model.output_shape, (64, 3))
        self.assertEqual(model.count_params(), 304223)
        self.assertEqual(len(model.layers), 58)


def get_input_shapes_3d():
    return {"input_A": (10, 9, 8, 1)}


def get_input_shapes_2d():
    return {"input_A": (10, 9, 1)}


def get_input_shapes_graph():
    return {"nodes": (20, 7), "is_valid": (20,), "coords": (20, 4)}
