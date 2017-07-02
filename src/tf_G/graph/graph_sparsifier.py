import tensorflow as tf

from tf_G.graph.graph import Graph


class GraphSparsifier(Graph):
    def __init__(self, sess: tf.Session, p: float, graph: Graph = None,
                 is_sparse: bool = False, name: str = None,
                 n: int = None, writer: tf.summary.FileWriter = None) -> None:

        self.p = p
        if graph is not None:

            if name is None:
                name = graph.name + "_sparsifier"

            edges_np = GraphSparsifier._sample_edges(sess, graph, p)

            Graph.__init__(self, sess, name, edges_np=edges_np, n=graph.n,
                           is_sparse=is_sparse, writer=writer)
        else:
            Graph.__init__(self, sess, name, n=n, is_sparse=is_sparse)

    @staticmethod
    def _sample_edges(sess: tf.Session, graph: Graph, p: float):

        distribution_tf = tf.random_uniform([graph.m], 0.0, 1.0)

        a = tf.reshape(tf.map_fn(
            lambda x: tf.gather(graph.out_degrees_tf, x),
            tf.slice(graph.edge_list_tf, [0, 0], [graph.m, 1]),
            dtype=tf.float32), [graph.m])

        cond_tf = p / tf.div(tf.log(tf.sqrt(graph.n_tf) + a),
                             tf.log(tf.sqrt(graph.n_tf)))

        return graph.edge_list_np[sess.run(
            tf.transpose(tf.less_equal(distribution_tf, cond_tf)))]

    def append(self, src: int, dst: int):
        distribution_tf = tf.random_uniform([1], 0.0, 1.0)

        cond_tf = self.p / tf.div(
            tf.log(tf.sqrt(self.n_tf) + self.out_degrees_tf_vertex(src)),
            tf.log(tf.sqrt(self.n_tf)))

        if self.run_tf(tf.less_equal(distribution_tf, cond_tf)):
            Graph.append(self, src, dst)
        print(self.m)

    def remove(self, src: int, dst: int):
        # TODO
        Graph.remove(self, src, dst)
