import tensorflow as tf
import numpy as np

from src.graph.graph import Graph


class GraphConstructor:
    @staticmethod
    def from_edges(sess, name, edges_np, writer=None):
        return Graph(sess, name, edges_np=edges_np, writer=writer)

    @staticmethod
    def empty(sess, name, n, writer=None):
        return Graph(sess, name, n=n, writer=writer)

    @staticmethod
    def random(sess, name, n, m, writer=None):
        # TODO working on
        edges_np = np.random.random_integers(0, n - 1, [m, 2])

        print(np.unique(edges_np))

        return Graph(sess, name, edges_np=edges_np, writer=writer)

    @staticmethod
    def as_naive_sparsifier(sess, graph, p):
        boolean_distribution = tf.less_equal(
            tf.random_uniform([graph.m], 0.0, 1.0), p)
        edges_np = graph.edge_list_np[sess.run(boolean_distribution)]
        return Graph(sess, graph.name + "_sparsifier",
                     edges_np=edges_np)

    @classmethod
    def as_other_sparsifier(cls, sess, graph, p):
        distribution_tf = tf.random_uniform([graph.m, 1], 0.0, 1.0)

        print(sess.run(distribution_tf))

        a = tf.map_fn(
            lambda x: tf.gather(graph.out_degrees_tf, tf.gather(x, 0)),
            graph.edge_list_tf,
            dtype=tf.float32)

        cond_tf = tf.map_fn(lambda x: (p / x if x is not 0 else p), a)
        print(sess.run(cond_tf))
        edges_np = graph.edge_list_np[sess.run(
            tf.gather(
                tf.transpose(tf.less_equal(distribution_tf, cond_tf)), 0))]
        print(edges_np.shape)

        return Graph(sess, graph.name + "_sparsifier", edges_np=edges_np)
