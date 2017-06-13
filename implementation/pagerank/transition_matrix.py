import tensorflow as tf


class TransitionMatrix:
    def __init__(self, sess, name, graph):
        self.sess = sess
        self.name = name
        self.G = graph
        self.transition = tf.Variable(tf.div(self.G.A_tf, self.G.E_o_degrees),
                                      name=self.name + "_T_naive")
        self.sess.run(tf.variables_initializer([self.transition]))

    @property
    def get(self):
        return self.transition
