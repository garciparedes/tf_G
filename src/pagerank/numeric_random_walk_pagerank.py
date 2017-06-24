import warnings

import tensorflow as tf
import numpy as np

from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.pagerank.transition_random import TransitionRandom
from src.utils.vector_convergence import VectorConvergenceCriterion


class NumericRandomWalkPageRank(NumericIterativePageRank):
    def __init__(self, sess, name, graph, beta=None):
        NumericIterativePageRank.__init__(self, sess, name, graph, beta)

        self.random_T = TransitionRandom(sess, name, self.T)

        self.iter = lambda t, v, n=self.G.n_tf: tf.add(
            tf.divide((v * t), tf.add(t, n)),
            tf.reshape(tf.py_func(self.proof, [n, t], tf.float32), shape=[1, 4])
        )

    def proof(self, n, t):
        v = np.zeros(shape=[1, 4], dtype='float32')
        selected = np.array([np.random.choice([1, 2, 3]),
                             np.random.choice([2, 3]),
                             np.random.choice([0]),
                             np.random.choice([0, 2])])
        for i in selected:
            v[0, i] += 1 / (n + t)
        return v
