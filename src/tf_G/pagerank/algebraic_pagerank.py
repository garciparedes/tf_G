import warnings
from typing import List

import tensorflow as tf

from tf_G.pagerank.transition.transition_matrix import TransitionMatrix
from tf_G.pagerank.pagerank import PageRank
from tf_G.graph.graph import Graph


class AlgebraicPageRank(PageRank):
  def __init__(self, sess: tf.Session, name: str, graph: Graph,
               beta: float) -> None:
    T = TransitionMatrix(sess, name + "_alge", graph)
    PageRank.__init__(self, sess, name, graph, beta, T)

  def _pr_exact_tf(self, topics: List[int] = None) -> tf.Tensor:
    if topics is not None:
      warnings.warn('Personalized PageRank not implemented yet!')
    a = tf.fill([1, self.G.n], (1 - self.beta) / self.G.n_tf)
    b = tf.matrix_inverse(
      tf.eye(self.G.n, self.G.n) - self.beta * self.T())
    self.run_tf(self.v.assign(tf.matmul(a, b)))
    return self.v

  def _pr_convergence_tf(self, convergence: float, topics: List[int] = None,
                         c_criterion=None) -> tf.Tensor:
    if topics is not None:
      warnings.warn('Personalized PageRank not implemented yet!')
    warnings.warn('PageRank not implements iterative PageRank! ' +
                  'Using exact algorithm.')
    return self._pr_exact_tf(topics)

  def _pr_steps_tf(self, steps: int, topics: List[int]) -> tf.Tensor:
    if topics is not None:
      warnings.warn('Personalized PageRank not implemented yet!')
    warnings.warn('PageRank not implements iterative PageRank! ' +
                  'Using exact algorithm.')
    return self._pr_exact_tf(topics)
