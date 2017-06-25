import warnings

import tensorflow as tf
import numpy as np

from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.pagerank.transition_random import TransitionRandom
from src.utils.vector_convergence import VectorConvergenceCriterion


class NumericRandomWalkPageRank(NumericIterativePageRank):
    def __init__(self, sess, name, graph, beta=None):
        NumericIterativePageRank.__init__(self, sess, name + "_rw", graph, beta)

        self.random_T = TransitionRandom(sess, self.name, self.T)

        self.iter = lambda t, v, n=self.G.n: tf.add(
            tf.divide((v * t), tf.add(t, n)),
            self.random_T(t))

    def _pr_convergence_tf(self, convergence, personalized=None,
                           convergence_criterion=VectorConvergenceCriterion.ONE):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        a = tf.while_loop(
            convergence_criterion,
            lambda i, v, v_last, c, n, dist: (
                i + 1,
                tf.add(
                    tf.divide(i * v, i + n),
                    tf.scatter_nd(
                        tf.reshape(dist, shape=[self.G.n, 1]),
                        tf.fill([self.G.n], 1 / (n + i)),
                        [self.G.n])),
                v,
                c,
                n,
                tf.cast(tf.squeeze(tf.multinomial(
                    tf.nn.embedding_lookup(self.random_T.T_log, dist),
                    num_samples=1)), tf.int32)
            ),
            [
                0.0,
                self.v,
                tf.zeros([1, self.G.n]),
                convergence,
                self.G.n_tf,
                tf.cast(tf.squeeze(tf.multinomial(
                    tf.nn.embedding_lookup(self.random_T.T_log,
                                           tf.range(0, self.G.n)),
                    num_samples=1)), tf.int32)
            ], name=self.name + "_while_conv")

        a_np = self.run(a)

        for array in a_np:
            print(array)
        print()
        self.run(self.v.assign(a_np[1]))
        return self.v

    def _pr_steps_tf(self, steps, personalized):
        if personalized is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        a = tf.while_loop(
            lambda i, v, dist: i < steps,
            lambda i, v, dist: (
                i + 1.0,
                tf.add(
                    tf.divide(i * v, i + self.G.n),
                    tf.scatter_nd(
                        tf.reshape(dist, shape=[self.G.n, 1]),
                        tf.fill([self.G.n], 1 / (self.G.n + i)),
                        [self.G.n])),
                tf.cast(tf.squeeze(tf.multinomial(
                    tf.nn.embedding_lookup(self.random_T.T_log, dist),
                    num_samples=1)), tf.int32)
            ),
            [
                0.0,
                self.v,
                tf.cast(tf.squeeze(tf.multinomial(
                    tf.nn.embedding_lookup(self.random_T.T_log,
                                           tf.range(0, self.G.n)),
                    num_samples=1)), tf.int32)
            ],
            name=self.name + "_while_steps")
        a_np = self.run(a)

        for array in a_np:
            print(array)
        print()
        self.run(self.v.assign(a_np[1]))
        return self.v
