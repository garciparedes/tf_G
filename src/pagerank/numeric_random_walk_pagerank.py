import tensorflow as tf
import numpy as np

from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.pagerank.transition_random import TransitionRandom


class NumericRandomWalkPageRank(NumericIterativePageRank):
    def __init__(self, sess, name, graph, beta=None):
        NumericIterativePageRank.__init__(self, sess, name, graph, beta)

        self.random_T = TransitionRandom(sess, name, self.T)
        self.t = tf.Variable(0.0)
        self.run(tf.variables_initializer([self.t]))

        self.iter = [self.v_last.assign(self.v),
                     self.t.assign_add(1),
                     self.v.assign(
                         tf.py_func(self.proof, [self.v], tf.float32) * self.t /
                         tf.add(self.t,
                                self.G.n_tf))]

    def proof(self, v):
        selected = np.array([np.random.choice([1, 2, 3]),
                             np.random.choice([2, 3]),
                             np.random.choice([0]),
                             np.random.choice([0, 2])])
        for i in selected:
            v[0,i] += 1 / self.run(self.G.n_tf + self.t)
        return v
