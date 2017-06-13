import tensorflow as tf

from pagerank.PageRank import PageRank
from utils import Utils


class NumericPageRank(PageRank):
    def __init__(self, sess, name, graph, reset_probability=None):
        super(NumericPageRank, self).__init__(sess)

        self.name = name
        self.G = graph

        if reset_probability is None:
            self.transition = tf.div(self.G.A_tf, self.G.E_o_degrees)
        else:
            beta = tf.constant(reset_probability, tf.float32, name="Beta")
            condition = tf.not_equal(self.G.E_o_degrees, 0)
            self.transition = tf.transpose(
                tf.where(condition,
                         tf.transpose(
                             beta * tf.div(self.G.A_tf, self.G.E_o_degrees) + (
                                 1 - beta) / self.G.n_tf),
                         tf.fill([self.G.n, self.G.n],
                                 tf.pow(self.G.n_tf, -1))))
        self.v_last = tf.Variable(tf.fill([self.G.n, 1], 0.0),
                                  name=name + "_Vi-1")
        self.v = tf.Variable(tf.fill([self.G.n, 1], tf.pow(self.G.n_tf, -1)),
                             name=name + "_Vi")
        self.sess.run(tf.variables_initializer([self.v_last, self.v]))

    def page_rank_vector(self, convergence=None, steps=None):
        page_rank = tf.matmul(self.transition, self.v, a_is_sparse=True)
        run_iteration = tf.assign(self.v, page_rank)
        if convergence is not None:
            diff = tf.reduce_max(tf.abs(tf.subtract(self.v_last, self.v)), 0)
            self.sess.run(tf.assign(self.v_last, self.v))
            self.sess.run(run_iteration)
            while self.sess.run(diff)[0] > convergence / self.sess.run(
                    self.G.n_tf):
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
