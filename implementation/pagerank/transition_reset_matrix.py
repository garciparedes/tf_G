import tensorflow as tf

from tensor_flow_object import TensorFlowObject


class TransitionResetMatrix(TensorFlowObject):
    def __init__(self, sess, name, graph, beta):
        TensorFlowObject.__init__(self, sess, name)
        self.G = graph

        self.transition = tf.Variable(
            tf.where(self.G.is_not_sink_vertice,
                     tf.add(
                         tf.scalar_mul(beta, tf.div(self.G.A_tf,
                                                    self.G.out_degrees)),
                         (1 - beta) / self.G.n_tf),
                     tf.fill([self.G.n, self.G.n], tf.pow(self.G.n_tf, -1))),
            name=self.name + "_T")
        self.run(tf.variables_initializer([self.transition]))

    @property
    def get(self):
        return self.transition
