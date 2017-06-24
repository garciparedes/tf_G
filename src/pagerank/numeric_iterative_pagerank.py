import warnings

import tensorflow as tf

from src.pagerank.numeric_pagerank import NumericPageRank
from src.pagerank.transition_reset_matrix import TransitionResetMatrix
from src.utils.vector_convergence import VectorConvergenceCriterion


class NumericIterativePageRank(NumericPageRank):
    def __init__(self, sess, name, graph, beta=None):
        T = TransitionResetMatrix(sess, name + "_iter", graph, beta)
        NumericPageRank.__init__(self, sess, name, graph, beta, T)
        self.iter = lambda i, a, b=self.T(): tf.matmul(a, b)

    def _pr_convergence_tf(self, convergence, personalized=None,
                           convergence_criterion=VectorConvergenceCriterion.ONE):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        self.run(
            self.v.assign(
                tf.while_loop(convergence_criterion,
                              lambda i, v, v_last, c, n:
                              (i + 1, self.iter(i, v), v, c, n),
                              [0.0, self.v, tf.zeros([1, self.G.n]),
                               convergence,
                               self.G.n_tf])[1]))
        return self.v

    def _pr_steps_tf(self, steps, personalized):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        self.run(
            self.v.assign(
                tf.while_loop(lambda i, v: i < steps,
                              lambda i, v: (i + 1.0, self.iter(i, v)),
                              [0.0, self.v])[1]))
        return self.v

    def _pr_exact_tf(self, personalized):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        raise NotImplementedError(
            str(self.__name__) + ' not implements exact PageRank')

    def update_edge(self, edge, change):
        warnings.warn('PageRank auto-update not implemented yet!')

        print("Edge: " + str(edge) + "\tChange: " + str(change))
        self.run(self._pr_convergence_tf(convergence=0.01))
