from graph.graph import Graph

import tensorflow as tf


class GraphSparsifier(Graph):
    def __init__(self, sess, name, edges_data_frame, p):
        self.boolean_distribution = tf.less_equal(
            tf.random_uniform([edges_data_frame.shape[0]],
                              0.0, 1.0), p)
        edges_data_frame = edges_data_frame[sess.run(self.boolean_distribution)]
        Graph.__init__(self, sess, name, edges_data_frame)
