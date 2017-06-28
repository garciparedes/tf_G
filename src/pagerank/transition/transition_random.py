import tensorflow as tf

from src.utils.tensorflow_object import TensorFlowObject


class TransitionRandom(TensorFlowObject):
    def __init__(self, sess, name, T, writer=None):
        TensorFlowObject.__init__(self, sess, name + "_T_log", writer=writer)
        self.T_log = tf.Variable(tf.log(T()), name=self.name)
        self.n = T.G.n
        self.run(tf.variables_initializer([self.T_log]))

    def __call__(self, *args, **kwargs):
        return self.get_tf(*args)

    def get_tf(self, t):
        return (tf.scatter_nd(
            tf.reshape(tf.multinomial(self.T_log, num_samples=1), [self.n, 1]),
            tf.fill([self.n], 1 / (self.n + t)),
            [self.n]))

    def update_edge(self, edge, change):
        # TODO
        pass
