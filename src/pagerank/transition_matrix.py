import tensorflow as tf

from src.tensor_flow_object import TensorFlowObject


class TransitionMatrix(TensorFlowObject):
    def __init__(self, sess, name, graph):
        TensorFlowObject.__init__(self, sess, name)
        self.G = graph
        self.G.attach(self)
        self.transition = tf.Variable(
            tf.where(self.G.is_not_sink_vertice,
                     tf.div(self.G.A_tf, self.G.out_degrees),
                     tf.fill([self.G.n, self.G.n], 0.0)),
            name=self.name + "_T")
        self.run(tf.variables_initializer([self.transition]))
        self.update_T = self.transition.assign(
            tf.where(self.G.is_not_sink_vertice,
                     tf.div(self.G.A_tf, self.G.out_degrees),
                     tf.fill([self.G.n, self.G.n], 0.0)))

    @property
    def get(self):
        return self.transition

    def update(self):
        self.run(self.update_T)
