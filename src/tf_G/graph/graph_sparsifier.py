import tensorflow as tf
import numpy as np
from tf_G.graph.graph import Graph


class GraphSparsifier(Graph):
    def __init__(self, sess: tf.Session, p: float, graph: Graph = None,
                 is_sparse: bool = False, name: str = None,
                 n: int = None, writer: tf.summary.FileWriter = None) -> None:
        if graph is not None:

            if name is None:
                name = graph.name + "_sparsifier"

            distribution_tf = tf.random_uniform([graph.m], 0.0, 1.0)

            a = tf.reshape(tf.map_fn(
                lambda x: tf.gather(graph.out_degrees_tf, x),
                tf.slice(graph.edge_list_tf, [0, 0], [graph.m, 1]),
                dtype=tf.float32), [graph.m])

            cond_tf = p / tf.div(tf.log(tf.sqrt(graph.n_tf) + a),
                                 tf.log(tf.sqrt(graph.n_tf)))

            edges_np = graph.edge_list_np[sess.run(
                tf.transpose(tf.less_equal(distribution_tf, cond_tf)))]
            # print(edges_np.shape)
            Graph.__init__(self, sess, name, edges_np=edges_np, n=graph.n,
                           is_sparse=is_sparse)
        else:
            pass

    def append(self, src: int, dst: int):
        Graph.append(self, src, dst)

    def remove(self, src: int, dst: int):
        Graph.remove(self, src, dst)
