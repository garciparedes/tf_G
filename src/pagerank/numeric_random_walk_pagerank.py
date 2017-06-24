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
        self.t = tf.Variable(0.0)
        self.run(tf.variables_initializer([self.t]))

        self.v_last = tf.Variable(tf.zeros([1, self.G.n]),
                                  name=self.G.name + "_" + self.name + "_Vi-1")
        self.run(tf.variables_initializer([self.v_last]))

        update_t = self.t.assign_add(1)

        with tf.control_dependencies([update_t]):
            update_v1 = self.v.assign(
                self.v.read_value() * self.t.read_value() / tf.add(
                    self.t.read_value(), self.G.n_tf.read_value()))

        with tf.control_dependencies([update_v1]):
            update_v2 = tf.assign_add(self.v, tf.py_func(self.proof,
                                                         [self.G.n_tf, self.t],
                                                         tf.float32))

        self.iter2 = [self.v_last.assign(self.v),
                      update_t,
                      update_v1,
                      update_v2]

    def _pr_convergence_tf(self, convergence, personalized=None,
                           convergence_criterion=VectorConvergenceCriterion.ONE):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        self.run(self.iter2)

        while self.run(convergence_criterion(self.v_last, self.v, convergence,
                                             self.G.n_tf)):
            self.run(self.iter2)

        return self.v

    def proof(self, n, t):
        v = np.zeros(shape=[1, 4], dtype='float32')
        selected = np.array([np.random.choice([1, 2, 3]),
                             np.random.choice([2, 3]),
                             np.random.choice([0]),
                             np.random.choice([0, 2])])
        for i in selected:
            v[0, i] = 1 / (n + t)
        return v
