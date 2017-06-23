import warnings

import tensorflow as tf

from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.pagerank.transition_random import TransitionRandom


class NumericRandomWalkPageRank(NumericIterativePageRank):
    def __init__(self, sess, name, graph, beta=None):
        NumericIterativePageRank.__init__(self, sess, name, graph, beta)

        self.random_T = TransitionRandom(sess, name, self.T)
        self.iter = [self.v_last.assign(self.v),
                     self.v.assign(tf.matmul(self.v, self.T.get_tf))]
        self.run(tf.variables_initializer([self.v_last]))
