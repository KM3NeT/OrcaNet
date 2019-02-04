from unittest import TestCase

from orcanet.model_archs import model_setup
from orcanet.core import Configuration

# filepath = '/home/woody/capn/mppi033h/Code/HPC/cnns/models/trained/trained_model_VGG_4d_xyz-t_and_yzt-x_muon-CC_to_elec-CC_double_input_single_train_epoch1.h5'
# trained_model = ks.models.load_model(filepath)
#
# dummy_input = np.ones((5,5))
#
# K.set_learning_phase(1)
# dropout_test = Dropout(0.3)
# out_1 = dropout_test.call(dummy_input)
# K.eval(out_1)
#
# dropout_test.rate = 0.5
# out_2 = dropout_test.call(dummy_input)
# K.eval(out_2)


class TestTest(TestCase):

    def test_true(self):
        self.assertTrue(True)

