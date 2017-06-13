import tensorflow as tf

from pagerank.page_rank import PageRank
from pagerank.transition_matrix import TransitionMatrix
from pagerank.transition_reset_matrix import TransitionResetMatrix
from utils import Utils
import numpy as np


class NumericPageRank(PageRank):
    def __init__(self, sess, name, graph, reset_probability=None):
        super(NumericPageRank, self).__init__(sess)
        self.name = name
        self.G = graph
        if reset_probability is None:
            self.transition = TransitionMatrix(self.G)
        else:
            self.transition = TransitionResetMatrix(self.G, reset_probability)
        self.v_last = tf.Variable(tf.fill([self.G.n, 1], 0.0),
                                  name=name + "_Vi-1")
        self.v = tf.Variable(tf.fill([self.G.n, 1], tf.pow(self.G.n_tf, -1)),
                             name=name + "_Vi")
        self.page_rank = tf.matmul(self.transition.get, self.v,
                                   a_is_sparse=True)
        self.iteration = tf.assign(self.v, self.page_rank)
        self.sess.run(tf.variables_initializer([self.v_last, self.v]))

    def page_rank_vector(self, convergence=None, steps=None):
        if convergence is not None:
            return self.page_rank_until_convergence(convergence)
        elif steps > 0:
            return self.page_rank_until_steps(steps)
        else:
            raise ValueError("'convergence' or 'steps' must be assigned")

    def page_rank_until_convergence(self, convergence):
        diff = tf.reduce_max(tf.abs(tf.subtract(self.v_last, self.v)), 0)
        self.sess.run(tf.assign(self.v_last, self.v))
        self.sess.run(self.iteration)
        while self.sess.run(diff)[0] > convergence / self.sess.run(
                self.G.n_tf):
            self.sess.run(tf.assign(self.v_last, self.v))
            self.sess.run(self.iteration)
        return self.sess.run(self.v)

    def page_rank_until_steps(self, steps):
        for step in range(steps):
            self.sess.run(self.iteration)
        return self.sess.run(self.v)

    def ranks(self, convergence=None, steps=None):
        if convergence or steps is not None:
            self.page_rank_vector(convergence, steps)
        ranks = tf.py_func(Utils.ranked, [tf.multiply(self.v, -1)], tf.int64)
        ranks = tf.map_fn(lambda x: [x, tf.gather(self.v, x)[0]], ranks,
                          dtype=[tf.int64, tf.float32])
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return np.concatenate(self.sess.run(ranks), axis=1)
