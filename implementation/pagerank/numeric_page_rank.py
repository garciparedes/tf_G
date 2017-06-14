import tensorflow as tf

from pagerank.page_rank import PageRank
from pagerank.transition_matrix import TransitionMatrix
from pagerank.transition_reset_matrix import TransitionResetMatrix
from utils import Utils
import numpy as np


class NumericPageRank(PageRank):
    def __init__(self, sess, name, graph, beta=None):
        PageRank.__init__(self, sess, name)
        self.G = graph
        if beta is None:
            self.T = TransitionMatrix(self.sess, self.name, self.G)
        else:
            self.T = TransitionResetMatrix(self.sess, self.name,
                                           self.G,
                                           beta)
        self.v = tf.Variable(tf.fill([1, self.G.n], tf.pow(self.G.n_tf, -1)),
                             name=self.name + "_Vi")
        self.page_rank = None
        self.sess.run(tf.variables_initializer([self.v]))

    def page_rank_until_convergence(self, convergence):
        raise NotImplementedError(
            'subclasses must override page_rank_until_convergence()!')

    def page_rank_until_steps(self, steps):
        raise NotImplementedError(
            'subclasses must override page_rank_until_steps()!')

    def ranks(self, convergence=None, steps=None):
        if convergence or steps is not None:
            self.page_rank_vector(convergence, steps)
        ranks = tf.transpose(
            tf.py_func(Utils.ranked, [tf.scalar_mul(-1, self.v)], tf.int64))
        ranks = tf.map_fn(lambda x: [x, tf.gather(tf.gather(self.v, 0), x)],
                          ranks,
                          dtype=[tf.int64, tf.float32])
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return np.concatenate(self.sess.run(ranks), axis=1)
