import tensorflow as tf
import warnings

from pagerank.numeric_page_rank import NumericPageRank
from pagerank.page_rank import PageRank
from pagerank.transition_matrix import TransitionMatrix
from utils import Utils
import numpy as np


class NumericAlgebraicPageRank(NumericPageRank):
    def __init__(self, sess, name, graph, beta):
        T = TransitionMatrix(sess, name, graph)

        NumericPageRank.__init__(self, sess, name, graph, beta, T)

    def _page_rank_exact(self):
        a = tf.fill([1, self.G.n], (1 - self.beta_tf) / self.G.n_tf)
        b = tf.matrix_inverse(
            tf.eye(self.G.n, self.G.n) - self.beta_tf * self.T.get)
        self.sess.run(self.v.assign(tf.matmul(a, b, b_is_sparse=True)))
        return self.sess.run(self.v)

    def _page_rank_until_convergence(self, convergence):
        warnings.warn('NumericPageRank not implements iterative PageRank! ' +
                      'Using exact algorithm.')
        return self._page_rank_exact()

    def _page_rank_until_steps(self, steps):
        warnings.warn('NumericPageRank not implements iterative PageRank! ' +
                      'Using exact algorithm.')
        return self._page_rank_exact()
