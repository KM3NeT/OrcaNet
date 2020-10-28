import numpy as np
import tensorflow as tf
import orcanet.lib.losses as on_losses


class TestLklNormal(tf.test.TestCase):
    def setUp(self):
        self.loss_func = on_losses.lkl_normal
        self.y_pred = tf.constant([
            [[0], [1]],
            [[1], [1]],
            [[2], [2]],
        ], dtype="float32")  # shape (3, 2, 1)
        self.y_true = tf.constant([
            [[0], [0]],
            [[0], [0]],
            [[0], [0]],
        ], dtype="float32")

        self.target = tf.constant([[0.9189385], [1.4189385], [2.1120858]])

    def test_normal(self):
        loss = self.loss_func(self.y_true, self.y_pred)
        self.assertAllClose(loss, self.target)

    def test_normal_flat(self):
        loss = self.loss_func(tf.squeeze(self.y_true), tf.squeeze(self.y_pred))
        self.assertAllClose(loss, tf.squeeze(self.target))

    def test_normal_double(self):
        y_pred = tf.repeat(self.y_pred, 4, -1)
        y_true = tf.repeat(self.y_true, 4, -1)
        loss = self.loss_func(y_true, y_pred)
        self.assertAllClose(loss, tf.repeat(self.target, 4, -1))

    def test_if_training_works(self):
        inp = tf.keras.Input((3,))
        x = tf.keras.layers.Dense(6)(inp)
        x = tf.keras.layers.Reshape((2, 3))(x)
        model = tf.keras.Model(inp, x)
        model.compile("sgd", loss=self.loss_func)

        xs = np.ones((5, 3))
        ys = np.ones((5, 2, 3))
        model.fit(xs, ys)


class TestLklNormalTfp(TestLklNormal):
    def setUp(self):
        super().setUp()
        self.loss_func = on_losses.lkl_normal_tfp
