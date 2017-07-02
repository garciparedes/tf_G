import warnings

import tensorflow as tf
import numpy as np

from typing import List

from tf_G import Graph
from tf_G.utils.utils import Utils
from tf_G.utils.vector_norm import VectorNorm
from tf_G.pagerank.transition import Transition
from tf_G.utils.tensorflow_object import TensorFlowObject
from tf_G.utils.convergence_criterion import ConvergenceCriterion


class PageRank(TensorFlowObject):
  def __init__(self, sess: tf.Session, name: str, graph: Graph, beta: float,
               T: Transition) -> None:
    TensorFlowObject.__init__(self, sess, name)
    self.G = graph
    self.beta = beta
    self.T = T
    self.T.attach(self)
    self.v = tf.Variable(tf.fill([1, self.G.n], tf.pow(self.G.n_tf, -1)),
                         name=self.G.name + "_" + self.name + "_v")
    self.run_tf(tf.variables_initializer([self.v]))

  def error_vector_compare_tf(self, other_pr: 'PageRank',
                              k: int = -1) -> tf.Tensor:
    if 0 < k < self.G.n - 1:
      if 0 < k < self.G.n - 1:
        warnings.warn('k-best error comparison not implemented yed')

    return tf.reshape(
      VectorNorm.ONE(tf.subtract(self.v, other_pr.v)), [])

  def error_vector_compare_np(self, other_pr: 'PageRank',
                              k: int = -1) -> np.array:
    return self.run_tf(self.error_vector_compare_tf(other_pr, k))

  def error_ranks_compare_tf(self, other_pr: 'PageRank',
                             k: int = -1) -> tf.Tensor:
    if 0 < k < self.G.n - 1:
      warnings.warn('k-best error comparison not implemented yed')

    return tf.div(tf.cast(tf.reduce_sum(tf.abs(
      tf.py_func(Utils.ranked, [
        tf.py_func(Utils.ranked, [tf.scalar_mul(-1, self.v)],
                   tf.int64)], tf.int64) -
      tf.py_func(Utils.ranked, [
        tf.py_func(Utils.ranked, [tf.scalar_mul(-1, other_pr.v)],
                   tf.int64)], tf.int64))), tf.float32),
      (self.G.n_tf * (self.G.n_tf - 1)))

  def error_ranks_compare_np(self, other_pr: 'PageRank',
                             k: int = -1) -> np.array:
    return self.run_tf(self.error_ranks_compare_tf(other_pr, k=k))

  def pagerank_vector_np(self, convergence: float = 1.0, steps: int = 0,
                         topics: List[int] = None,
                         c_criterion=ConvergenceCriterion.ONE) -> np.array:
    return self.run_tf(
      self.pagerank_vector_tf(convergence, steps, topics,
                              c_criterion))

  def pagerank_vector_tf(self, convergence: float = 1.0, steps: int = 0,
                         topics: List[int] = None,
                         c_criterion=ConvergenceCriterion.ONE) -> tf.Tensor:
    if 0.0 < convergence < 1.0:
      return self._pr_convergence_tf(convergence, topics=topics,
                                     c_criterion=c_criterion)
    elif steps > 0:
      return self._pr_steps_tf(steps, topics=topics)
    else:
      return self._pr_exact_tf(topics=topics)

  def ranks_np(self, convergence: float = 1.0, steps: int = 0,
               topics: List[int] = None) -> np.array:
    self.pagerank_vector_tf(convergence, steps, topics)
    ranks = tf.map_fn(
      lambda x: [x, tf.gather(tf.reshape(self.v, [self.G.n]), x)],
      tf.transpose(
        tf.py_func(Utils.ranked,
                   [tf.scalar_mul(-1, self.v)], tf.int64)),
      dtype=[tf.int64, tf.float32])
    return np.concatenate(self.run_tf(ranks), axis=1)

  def _pr_convergence_tf(self, convergence: float, topics: List[int],
                         c_criterion) -> tf.Tensor:
    raise NotImplementedError(
      'subclasses must override page_rank_until_convergence()!')

  def _pr_steps_tf(self, steps: int, topics: List[int]) -> tf.Tensor:
    raise NotImplementedError(
      'subclasses must override page_rank_until_steps()!')

  def _pr_exact_tf(self, topics: List[int]) -> tf.Tensor:
    raise NotImplementedError(
      'subclasses must override page_rank_exact()!')

  def update_edge(self, edge, change):
    warnings.warn('PageRank auto-update not implemented yet!')
