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
            self.random_T.get_tf(n,t, self.G.n))