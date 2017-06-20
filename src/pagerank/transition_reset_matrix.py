import warnings
import tensorflow as tf

from src.utils.tensorflow_object import TensorFlowObject


class TransitionResetMatrix(TensorFlowObject):
    def __init__(self, sess, name, graph, beta):
        TensorFlowObject.__init__(self, sess, name)

        self.G = graph
        self.G.attach(self)

        self.beta = beta
        self.transition = tf.Variable(
            tf.where(self.G.is_not_sink_tf,
                     tf.add(
                         tf.scalar_mul(beta, tf.div(self.G.A_tf,
                                                    self.G.out_degrees_tf)),
                         (1 - beta) / self.G.n_tf),
                     tf.fill([self.G.n, self.G.n], tf.pow(self.G.n_tf, -1))),
            name=self.name + "_T")
        self.run(tf.variables_initializer([self.transition]))

    @property
    def get_tf(self):
        return self.transition

    def update(self, edge, change):
        warnings.warn('TransitionResetMatrix auto-update not implemented yet!')

        print("Edge: " + str(edge) + "\tChange: " + str(change))

        self.run(tf.scatter_nd_update(
            self.transition, [[edge[0]]],
            tf.gather(
                tf.where(self.G.is_not_sink_tf,
                         tf.add(
                             tf.scalar_mul(
                                 self.beta,
                                 tf.div(
                                     self.G.A_tf,
                                     self.G.out_degrees_tf)),
                             (
                                 1 - self.beta) / self.G.n_tf),
                         tf.fill([self.G.n,
                                  self.G.n],
                                 tf.pow(
                                     self.G.n_tf,
                                     -1))),
                [edge[0]])))

