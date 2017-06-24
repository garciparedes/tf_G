import tensorflow as tf

from src.utils.tensorflow_object import TensorFlowObject


class TransitionRandom(TensorFlowObject):
    def __init__(self, sess, name, T, writer=None):
        TensorFlowObject.__init__(self, sess, name + "_T_log", writer=writer)
        self.T_log = tf.Variable(tf.log(T()), name=self.name)
        self.run(tf.variables_initializer([self.T_log]))

    def __call__(self, *args, **kwargs):
        return self.get_tf(*args)

    def get_tf(self, n, t):
        a = tf.scatter_nd(
            tf.reshape(tf.multinomial(self.T_log, num_samples=1), [n, 1]),
            tf.fill([n], 1 / (n + t)),
            [n])
        return a
