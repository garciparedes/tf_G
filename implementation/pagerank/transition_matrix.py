import tensorflow as tf

from tensor_flow_object import TensorFlowObject


class TransitionMatrix(TensorFlowObject):
    def __init__(self, sess, name, graph):
        TensorFlowObject.__init__(self, sess, name)
        self.G = graph
        self.transition = tf.Variable(
            tf.where(self.G.is_not_sink_vertice,
                     tf.div(self.G.A_tf, self.G.E_o_degrees),
                     tf.fill([self.G.n, self.G.n], 0.0)),
            name=self.name + "_T_naive")
        self.sess.run(tf.variables_initializer([self.transition]))

    @property
    def get(self):
        return self.transition
