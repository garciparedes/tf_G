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
        data_tf = tf.random_uniform([graph.m, 1], 0.0, 1.0)
        data_tf = tf.concat((graph.edge_list_np, data_tf), axis=1)

        print(sess.run(data_tf))

        data_tf = tf.map_fn(
            lambda x: (
                x[2] / tf.gather(graph.out_degrees_tf, tf.cast(tf.gather(x, 0),
                                                               tf.int64)) if tf.gather(
                    graph.out_degrees_tf,
                    tf.cast(tf.gather(x, 0), tf.int64)) is not 0 else x),
            data_tf, dtype=tf.float32)

        print(sess.run(data_tf))

        data_tf = tf.less_equal(
            data_tf, p)
        print(sess.run(data_tf))
        print(graph.edge_list_np)
        print(sess.run(graph.out_degrees_tf))
        pass