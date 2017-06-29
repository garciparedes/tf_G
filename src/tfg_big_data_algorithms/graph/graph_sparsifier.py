import tensorflow as tf

from tfg_big_data_algorithms.graph.graph import Graph


class GraphSparsifier(Graph):
    def __init__(self, sess: tf.Session, graph: Graph, p: float,
                 is_sparse: bool = False) -> None:
        distribution_tf = tf.random_uniform([graph.m], 0.0, 1.0)

        v = tf.Variable(graph.out_degrees_tf)
        sess.run(tf.variables_initializer([v]))

        # print(sess.run(distribution_tf))
        a = tf.reshape(tf.map_fn(
            lambda x: tf.gather(v, x),
            tf.slice(graph.edge_list_tf, [0, 0], [graph.m, 1]),
            dtype=tf.float32), [graph.m])

        cond_tf = p / tf.div(tf.log(tf.sqrt(graph.n_tf) + a),
                             tf.log(tf.sqrt(graph.n_tf)))

        edges_np = graph.edge_list_np[sess.run(
            tf.transpose(tf.less_equal(distribution_tf, cond_tf)))]
        # print(edges_np.shape)

        Graph.__init__(self, sess, graph.name + "_sparsifier",
                       edges_np=edges_np,
                       is_sparse=is_sparse)
