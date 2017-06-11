import tensorflow as tf

from pagerank.PageRank import PageRank
from utils import Utils


class NumericNaivePageRank(PageRank):
    def __init__(self, sess, graph_edges):
        super(NumericNaivePageRank, self).__init__(sess)

        n_raw = graph_edges.max(axis=0).max() + 1

        self.n = tf.constant(n_raw, tf.float32, name="NodeCounts")
        self.a = tf.Variable(tf.transpose(
            tf.scatter_nd(graph_edges.values.tolist(),
                          graph_edges.shape[0] * [1.0],
                          [n_raw, n_raw])), tf.float64, name="AdjacencyMatrix")
        self.v = tf.Variable(tf.fill([n_raw, 1], tf.pow(self.n, -1)),
                             name="PageRankVector")

        o_degree = tf.reduce_sum(self.a, 0)

        self.transition = tf.div(self.a, o_degree)

    def page_rank_vector(self, steps=10):
        page_rank = tf.matmul(self.transition, self.v, a_is_sparse=True)

        run_iteration = tf.assign(self.v, page_rank)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for step in range(steps):
            self.sess.run(run_iteration)
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return self.sess.run(self.v)

    def ranks(self):
        ranks = tf.transpose(
            tf.py_func(Utils.ranked, [tf.multiply(self.v,-1)], tf.int64))[0]
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return self.sess.run(ranks)
