import tensorflow as tf

from pagerank.PageRank import PageRank
from utils import Utils


class NumericPageRank(PageRank):
    def __init__(self, sess, graph_edges, reset_probability=None):
        super(NumericPageRank, self).__init__(sess)

        n_raw = graph_edges.max(axis=0).max() + 1
        self.n = tf.constant(n_raw, tf.float32, name="NodeCounts")

        a = tf.Variable(tf.transpose(
            tf.scatter_nd(graph_edges.values.tolist(),
                          graph_edges.shape[0] * [1.0],
                          [n_raw, n_raw])), tf.float64, name="AdjacencyMatrix")
        o_degree = tf.reduce_sum(a, 0)

        if reset_probability is None:
            self.transition = tf.div(a, o_degree)
        else:
            beta = tf.constant(reset_probability, tf.float32, name="Beta")
            condition = tf.not_equal(o_degree, 0)
            self.transition = tf.transpose(
                tf.where(condition,
                         tf.transpose(beta * tf.div(a, o_degree) + (
                             1 - beta) / self.n),
                         tf.fill([n_raw, n_raw], tf.pow(self.n, -1))))
        self.v_last = tf.Variable(tf.fill([n_raw, 1], 0.0),
                                  name="PageRankVectorLast")
        self.v = tf.Variable(tf.fill([n_raw, 1], tf.pow(self.n, -1)),
                             name="PageRankVector")
        self.sess.run(tf.global_variables_initializer())

    def page_rank_vector(self, convergence=None, steps=None):
        page_rank = tf.matmul(self.transition, self.v, a_is_sparse=True)
        run_iteration = tf.assign(self.v, page_rank)
        if convergence is not None:
            diff = tf.reduce_max(tf.abs(tf.subtract(self.v_last, self.v)), 0)
            self.sess.run(tf.assign(self.v_last, self.v))
            self.sess.run(run_iteration)
            while self.sess.run(diff)[0] > convergence / self.sess.run(self.n):
                self.sess.run(tf.assign(self.v_last, self.v))
                self.sess.run(run_iteration)
        elif steps > 0:
            for step in range(steps):
                self.sess.run(run_iteration)
        else:
            raise ValueError("'convergence' or 'steps' must be assigned")
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return self.sess.run(self.v)

    def ranks(self):
        ranks = tf.py_func(Utils.ranked, [tf.multiply(self.v, -1)], tf.int64)
        ranks = tf.map_fn(lambda x: [x, tf.gather(self.v, x)[0]], ranks,
                          dtype=[tf.int64, tf.float32])
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return self.sess.run(ranks)
