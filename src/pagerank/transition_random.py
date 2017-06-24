import tensorflow as tf

from src.utils.tensorflow_object import TensorFlowObject


class TransitionRandom(TensorFlowObject):
    def __init__(self, sess, name, T, writer=None):
        TensorFlowObject.__init__(self, sess, name, writer=writer)
        self.T = T

    def get_tf(self, n_tf, t, n):
        a = tf.scatter_nd(
            tf.reshape(tf.multinomial(tf.log(self.T.get_tf), num_samples=1),
                       [n, 1]),
            tf.fill([n], 1 / (n_tf + t)), [n])
        return a
