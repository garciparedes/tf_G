from graph.graph import Graph

import tensorflow as tf


class GraphSparsifier(Graph):
    def __init__(self, sess, name, edges_np, n, p):
        self.boolean_distribution = tf.less_equal(
            tf.random_uniform([edges_np.shape[0]],
                              0.0, 1.0), p)
        edges_np = edges_np[sess.run(self.boolean_distribution)]

        Graph.__init__(self, sess, name, edges_np=edges_np, n=n)
