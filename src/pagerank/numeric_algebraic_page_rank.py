import warnings

import tensorflow as tf

from src.pagerank.numeric_page_rank import NumericPageRank
from src.pagerank.transition_matrix import TransitionMatrix


class NumericAlgebraicPageRank(NumericPageRank):
    def __init__(self, sess, name, graph, beta):
        T = TransitionMatrix(sess, name, graph)

        NumericPageRank.__init__(self, sess, name, graph, beta, T)

    def _page_rank_exact(self, personalized=None):
        if personalized:
            pass
        else:
            pass
        a = tf.fill([1, self.G.n], (1 - self.beta) / self.G.n_tf)
        b = tf.matrix_inverse(
            tf.eye(self.G.n, self.G.n) - self.beta * self.T.get)
        self.run(self.v.assign(tf.matmul(a, b, b_is_sparse=True)))
        return self.run(self.v)

    def _page_rank_until_convergence(self, convergence, personalized=None):
        warnings.warn('NumericPageRank not implements iterative PageRank! ' +
                      'Using exact algorithm.')
        return self._page_rank_exact(personalized)

    def _page_rank_until_steps(self, steps, personalized=None):
        warnings.warn('NumericPageRank not implements iterative PageRank! ' +
                      'Using exact algorithm.')
        return self._page_rank_exact(personalized)
