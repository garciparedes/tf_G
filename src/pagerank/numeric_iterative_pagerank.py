import warnings

import tensorflow as tf

from src.pagerank.numeric_pagerank import NumericPageRank
from src.pagerank.transition_reset_matrix import TransitionResetMatrix
from src.utils.vector_convergence import VectorConvergenceCriterion


class NumericIterativePageRank(NumericPageRank):
    def __init__(self, sess, name, graph, beta=None):
        T = TransitionResetMatrix(sess, name, graph, beta)
        NumericPageRank.__init__(self, sess, name, graph, beta, T)
        self.v_last = tf.Variable(tf.zeros([1, self.G.n]),
                                  name=self.G.name + "_" + self.name + "_Vi-1")
        self.iter = [self.v_last.assign(self.v),
                     self.v.assign(tf.matmul(self.v, self.T.get_tf))]
        self.run(tf.variables_initializer([self.v_last]))

    def _pr_convergence_tf(self, convergence, personalized=None,
                           convergence_criterion=VectorConvergenceCriterion.ONE):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        self.run(self.iter)
        while self.run(convergence_criterion(self.v_last, self.v, convergence,
                                             self.G.n_tf)):
            self.run(self.iter)
        return self.v

    def _pr_steps_tf(self, steps, personalized):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        for step in range(steps):
            self.run(self.iter)
        return self.v

    def _pr_exact_tf(self, personalized):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        raise NotImplementedError(
            'NumericIterativePageRank not implements exact PageRank')

    def update_edge(self, edge, change):
        warnings.warn('PageRank auto-update not implemented yet!')

        print("Edge: " + str(edge) + "\tChange: " + str(change))
        self.run(self._pr_convergence_tf(convergence=0.01))
