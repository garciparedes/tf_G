import tensorflow as tf

from pagerank.transition_matrix import TransitionMatrix


class TransitionResetMatrix(TransitionMatrix):
    def __init__(self, sess, name, graph, beta):
        TransitionMatrix.__init__(self, sess, name, graph)
        self.transition = tf.Variable(
            tf.where(self.G.is_not_sink_vertice,
                     tf.add(
                         tf.scalar_mul(beta, tf.div(self.G.A_tf,
                                                    self.G.E_o_degrees)),
                         (1 - beta) / self.G.n_tf),
                     tf.fill([self.G.n, self.G.n], tf.pow(self.G.n_tf, -1))),
            name=self.name + "_T_reset")
        self.sess.run(tf.variables_initializer([self.transition]))

    @property
    def get(self):
        return self.transition
