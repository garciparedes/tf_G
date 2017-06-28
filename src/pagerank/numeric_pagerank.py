import warnings
from typing import List

import numpy as np
import tensorflow as tf

from pagerank.transition.transition import Transition
from src.graph.graph import Graph
from src.pagerank.pagerank import PageRank
from src.utils.utils import Utils
from src.utils.vector_norm import VectorNorm


class NumericPageRank(PageRank):
    def __init__(self, sess: tf.Session, name: str, graph: Graph, beta: float,
                 T: Transition) -> None:
        PageRank.__init__(self, sess, name)
        self.G = graph

        self.beta = beta
        self.T = T
        self.T.attach(self)
        self.v = tf.Variable(tf.fill([1, self.G.n], tf.pow(self.G.n_tf, -1)),
                             name=self.G.name + "_" + self.name + "_v")
        self.run(tf.variables_initializer([self.v]))

    def ranks_np(self, convergence: float = 1.0, steps: int = 0,
                 topics: List[int] = None) -> np.ndarray:
        self.pagerank_vector_tf(convergence, steps, topics)
        ranks = tf.map_fn(
            lambda x: [x, tf.gather(tf.reshape(self.v, [self.G.n]), x)],
            tf.transpose(
                tf.py_func(Utils.ranked,
                           [tf.scalar_mul(-1, self.v)], tf.int64)),
            dtype=[tf.int64, tf.float32])
        return np.concatenate(self.run(ranks), axis=1)

    def error_vector_compare_tf(self, other_pr: 'NumericPageRank',
                                k: int = -1) -> tf.Tensor:
        if 0 < k < self.G.n - 1:
            if 0 < k < self.G.n - 1:
                warnings.warn('k-best error comparison not implemented yed')

        return tf.reshape(
            VectorNorm.ONE(tf.subtract(self.v, other_pr.v)), [])

    def error_ranks_compare_tf(self, other_pr: 'NumericPageRank',
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
