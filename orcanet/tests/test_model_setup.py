# removed, as testing whether changing dropout works requires
# get layer output, which doesn't work properly in tf 2.1


import unittest
import numpy as np
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from orcanet.model_builder import _change_dropout_rate


@unittest.skip("skipped, as testing whether changing dropout works "
               "requires get layer output, which doesn't work "
               "properly in tf 2.1")
class TestDropoutChange(unittest.TestCase):
    def setUp(self):
        def dropout_model(rate_before, rate_after):
            inp1 = layers.Input((5, 1))
            x1 = layers.Dropout(rate_before)(inp1)
            x1 = layers.Dense(5)(x1)

            inp2 = layers.Input((5, 1))
            x2 = layers.Dropout(rate_before)(inp2)

            x = layers.Concatenate(axis=-1)([x1, x2])
            x = layers.Dense(5)(x)
            out = layers.Dropout(rate_after)(x)

            model = ks.models.Model([inp1, inp2], out)
            return model

        def get_layer_output(model, samples, layer_no=-1):
            l_out = ks.backend.function(
                model.input + [ks.backend.learning_phase(), ],
                [model.layers[layer_no].output])
            # output in train mode = 1
            layer_output = l_out(samples + [1, ])[0]
            return layer_output

        def calculate_rate(model, samples, layer_no):
            layer_output = get_layer_output(model, samples, layer_no)
            rate = np.sum(layer_output == 0)/layer_output.size
            return rate

        self.calculate_rate = calculate_rate
        self.concat_layer_no = 6

        self.model_0 = dropout_model(0., 0.)
        self.xs = [np.ones((50, 5, 1)), np.ones((50, 5, 1))]

    def test_change_dropout_after_concat(self):
        model_changed = _change_dropout_rate(self.model_0,
                                            before_concat=0.0,
                                            after_concat=0.999)
        rate_before_conc = self.calculate_rate(model_changed, self.xs,
                                               self.concat_layer_no)
        rate_after_conc = self.calculate_rate(model_changed, self.xs, -1)
        print("rate before Concatenate: {}\tafter: {}".format(rate_before_conc,
                                                              rate_after_conc))
        self.assertLess(rate_before_conc, 0.2)
        self.assertGreater(rate_after_conc, 0.6)

    def test_change_dropout_before_and_after_concat(self):
        model_changed = _change_dropout_rate(self.model_0,
                                            before_concat=0.999,
                                            after_concat=0.999)
        rate_before_conc = self.calculate_rate(model_changed, self.xs,
                                               self.concat_layer_no)
        rate_after_conc = self.calculate_rate(model_changed, self.xs, -1)
        print("rate before Concatenate: {}\tafter: {}".format(rate_before_conc,
                                                              rate_after_conc))
        self.assertGreater(rate_before_conc, 0.6)
        self.assertGreater(rate_after_conc, 0.6)

    def test_weights_are_copied_over(self):
        model_changed = _change_dropout_rate(self.model_0,
                                            before_concat=0.999,
                                            after_concat=0.999)
        for layer_no in range(len(self.model_0.layers)):
            weights_0 = self.model_0.layers[layer_no].get_weights()
            weights_changed = model_changed.layers[layer_no].get_weights()
            for i in range(len(weights_0)):
                self.assertTrue(np.array_equal(weights_0[i], weights_changed[i]))
