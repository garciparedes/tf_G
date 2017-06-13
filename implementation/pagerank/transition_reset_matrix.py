import tensorflow as tf

from pagerank.transition_matrix import TransitionMatrix


class TransitionResetMatrix(TransitionMatrix):
    def __init__(self, graph, reset_probability):
        super(TransitionResetMatrix, self).__init__(graph)
        self.beta = reset_probability
        self.beta_tf = tf.constant(self.beta, tf.float32, name="Beta")

    @property
    def get(self):
        condition = tf.not_equal(self.G.E_o_degrees, 0)
        return tf.transpose(
            tf.where(condition,
                     tf.transpose(self.beta_tf *
                                  tf.div(self.G.A_tf, self.G.E_o_degrees) + (
                                      1 - self.beta_tf) / self.G.n_tf),
                     tf.fill([self.G.n, self.G.n],
                             tf.pow(self.G.n_tf, -1))))
