import warnings
from typing import List

import tensorflow as tf
import numpy as np

from tf_G.pagerank.transition.transition_reset_matrix import \
    TransitionResetMatrix
from tf_G.pagerank.pagerank import PageRank
from tf_G.utils.convergence_criterion import ConvergenceCriterion
from tf_G.graph.graph import Graph


class IterativePageRank(PageRank):
    def __init__(self, sess: tf.Session, name: str, graph: Graph,
                 beta: float) -> None:
        T = TransitionResetMatrix(sess, name + "_iter", graph, beta)
        PageRank.__init__(self, sess, name + "_iter", graph, beta, T)
        self.iter = lambda i, a, b=self.T(): tf.matmul(a, b)

    def _pr_convergence_tf(self, convergence: float, topics: List[int] = None,
                           c_criterion=ConvergenceCriterion.ONE) -> tf.Tensor:
        if topics is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        self.run_tf(
            self.v.assign(
                tf.while_loop(c_criterion,
                              lambda i, v, v_last, c, n:
                              (i + 1, self.iter(i, v), v, c, n),
                              [0.0, self.v, tf.zeros([1, self.G.n]),
                               convergence,
                               self.G.n_tf], name=self.name + "_while_conv")[
                    1]))
        return self.v

    def _pr_steps_tf(self, steps: int, topics: List[int]) -> tf.Tensor:
        if topics is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        self.run_tf(
            self.v.assign(
                tf.while_loop(lambda i, v: i < steps,
                              lambda i, v: (i + 1.0, self.iter(i, v)),
                              [0.0, self.v], name=self.name + "_while_steps")[
                    1]))
        return self.v

    def _pr_exact_tf(self, topics: List[int]) -> None:
        if topics is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        raise NotImplementedError(
            str(self.__class__.__name__) + ' not implements exact PageRank')

    def update_edge(self, edge: np.array, change: float) -> None:
        self.run_tf(self._pr_convergence_tf(convergence=0.01))
