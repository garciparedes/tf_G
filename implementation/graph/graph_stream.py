from datasets import DataSets
from graph.graph import Graph

import tensorflow as tf


class GraphStream(Graph):
    def __init__(self, sess, name):
        Graph.__init__(self, sess, name, DataSets.empty())

    def append(self, src, dst):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")

        self.edges_df = self.edges_df.append({'src': src, 'dst': dst},
                                             ignore_index=True)
        self.sess.run(tf.assign(self.A_tf,
                                tf.scatter_nd(self.E_list, self.m * [1.0],
                                              [self.n, self.n]),
                                validate_shape=False))

    def remove(self, src=None, dst=None):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.edges_df = self.edges_df[(self.edges_df['src'] != src) &
                                      (self.edges_df['dst'] != dst)]
        self.sess.run(tf.assign(self.A_tf,
                                tf.scatter_nd(self.E_list, self.m * [1.0],
                                              [self.n, self.n]),
                                validate_shape=False))
