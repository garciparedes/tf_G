import tensorflow as tf

from pagerank.transition_matrix import TransitionMatrix


class TransitionResetMatrix(TransitionMatrix):
    def __init__(self, sess, name, graph, reset_probability):
        super(TransitionResetMatrix, self).__init__(sess, name, graph)
        self.beta = reset_probability
        self.beta_tf = tf.constant(self.beta, tf.float32, name="Beta")
        condition = tf.not_equal(self.G.E_o_degrees, 0)

        self.transition = tf.Variable(tf.transpose(
            tf.where(condition,
                     tf.transpose(self.beta_tf *
                                  tf.div(self.G.A_tf, self.G.E_o_degrees) + (
                                      1 - self.beta_tf) / self.G.n_tf),
                     tf.fill([self.G.n, self.G.n],
                             tf.pow(self.G.n_tf, -1)))), name=self.name + "_T_reset")

        self.sess.run(tf.variables_initializer([self.transition])) \

@ property


def get(self):
    return self.transition
