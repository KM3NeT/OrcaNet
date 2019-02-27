from unittest import TestCase
from keras.models import Model
from keras.layers import Input, Concatenate, Dropout, Dense
import numpy as np
from keras import backend as K

from orcanet.model_builder import change_dropout_rate


class TestDropoutChange(TestCase):

    def setUp(self):
        def dropout_model(rate_before, rate_after):
            inp1 = Input((5, 1))
            x1 = Dropout(rate_before)(inp1)
            x1 = Dense(5)(x1)

            inp2 = Input((5, 1))
            x2 = Dropout(rate_before)(inp2)

            x = Concatenate(axis=-1)([x1, x2])
            x = Dense(5)(x)
            out = Dropout(rate_after)(x)

            model = Model([inp1, inp2], out)
            return model

        def get_layer_output(model, samples, layer_no=-1):
            l_out = K.function(model.input + [K.learning_phase(), ],
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
        model_changed = change_dropout_rate(self.model_0,
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
        model_changed = change_dropout_rate(self.model_0,
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
        model_changed = change_dropout_rate(self.model_0,
                                            before_concat=0.999,
                                            after_concat=0.999)
        for layer_no in range(len(self.model_0.layers)):
            weights_0 = self.model_0.layers[layer_no].get_weights()
            weights_changed = model_changed.layers[layer_no].get_weights()
            for i in range(len(weights_0)):
                self.assertTrue(np.array_equal(weights_0[i], weights_changed[i]))
