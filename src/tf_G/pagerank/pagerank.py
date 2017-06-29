import warnings

import tensorflow as tf
import numpy as np

from typing import List
from tf_G.utils.tensorflow_object import TensorFlowObject
from tf_G.utils.vector_convergence import \
    ConvergenceCriterion


class PageRank(TensorFlowObject):
    def __init__(self, sess: tf.Session, name: str) -> None:
        TensorFlowObject.__init__(self, sess, name)

    def error_vector_compare_tf(self, other_pr: 'PageRank',
                                k: int = -1) -> tf.Tensor:
        raise NotImplementedError(
            'subclasses must override compare()!')

    def error_vector_compare_np(self, other_pr: 'PageRank',
                                k: int = -1) -> np.ndarray:
        return self.run(self.error_vector_compare_tf(other_pr, k))

    def error_ranks_compare_tf(self, other_pr: 'PageRank',
                               k: int = -1) -> np.ndarray:
        raise NotImplementedError(
            'subclasses must override compare()!')

    def error_ranks_compare_np(self, other_pr: 'PageRank',
                               k: int = -1) -> np.ndarray:
        return self.run(self.error_ranks_compare_tf(other_pr, k=k))

    def pagerank_vector_np(self, convergence: float = 1.0, steps: int = 0,
                           topics: List[int] = None,
                           c_criterion=ConvergenceCriterion.ONE) -> np.ndarray:
        return self.run(
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
                 topics: List[int] = None) -> np.ndarray:
        raise NotImplementedError(
            'subclasses must override ranks_by_rank()!')

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
