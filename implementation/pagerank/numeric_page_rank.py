import tensorflow as tf
import warnings

from pagerank.page_rank import PageRank
from pagerank.transition_matrix import TransitionMatrix
from utils import Utils
import numpy as np


class NumericPageRank(PageRank):
    def __init__(self, sess, name, graph, beta, T):
        PageRank.__init__(self, sess, name)
        self.G = graph
        self.beta = beta
        self.T = T
        self.v = tf.Variable(tf.fill([1, self.G.n], tf.pow(self.G.n_tf, -1)),
                             name=self.name + "_Vi")
        self.sess.run(tf.variables_initializer([self.v]))

    def ranks(self, convergence=1.0, steps=0):
        self.page_rank_vector(convergence, steps)
        ranks = tf.transpose(
            tf.py_func(Utils.ranked, [tf.scalar_mul(-1, self.v)], tf.int64))
        ranks = tf.map_fn(lambda x: [x, tf.gather(tf.gather(self.v, 0), x)],
                          ranks,
                          dtype=[tf.int64, tf.float32])
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return np.concatenate(self.sess.run(ranks), axis=1)
