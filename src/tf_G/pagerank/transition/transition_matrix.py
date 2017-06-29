import tensorflow as tf
import numpy as np

from tf_G.pagerank.transition.transition import Transition
from tf_G.graph.graph import Graph


class TransitionMatrix(Transition):
    def __init__(self, sess: tf.Session, name: str, graph: Graph) -> None:
        Transition.__init__(self, sess, name, graph)

        self.transition = tf.Variable(
            tf.where(self.G.is_not_sink_tf,
                     tf.div(self.G.A_tf, self.G.out_degrees_tf),
                     tf.fill([self.G.n, self.G.n], 1 / self.G.n)),
            name=self.name)
        self.run(tf.variables_initializer([self.transition]))

    def __call__(self, *args, **kwargs):
        return self.transition

    def update_edge(self, edge: np.ndarray, change: float) -> None:

        # print("Edge: " + str(edge) + "\tChange: " + str(change))

        if change > 0.0:
            self.run(tf.scatter_nd_update(
                self.transition, [[edge[0]]],
                tf.div(self.G.A_tf_vertex(edge[0]),
                       self.G.out_degrees_tf_vertex(edge[0]))))
        else:
            self.run(tf.scatter_nd_update(
                self.transition, [[edge[0]]],
                tf.where(self.G.is_not_sink_tf_vertex(edge[0]),
                         tf.div(self.G.A_tf_vertex(edge[0]),
                                self.G.out_degrees_tf_vertex(edge[0])),
                         tf.fill([1, self.G.n], tf.pow(self.G.n_tf, -1)))))
        self._notify(edge, change)
