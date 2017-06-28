import math
import warnings

import tensorflow as tf

from pagerank.transition.transition_random import TransitionRandom
from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.utils.vector_convergence import ConvergenceCriterion


class NumericRandomWalkPageRank(NumericIterativePageRank):
    def __init__(self, sess, name, graph, beta=None):
        warnings.warn('This implementation is in alpha. ' +
                      'Use NumericIterativePageRank Implementation!')

        NumericIterativePageRank.__init__(self, sess, name + "_rw", graph, beta)

        self.random_T = TransitionRandom(sess, self.name, self.T)

        self.iter = lambda t, v, n=self.G.n: tf.add(
            tf.divide((v * t), tf.add(t, n)),
            self.random_T(t))

    def _pr_convergence_tf(self, convergence, topics=None,
                           c_criterion=ConvergenceCriterion.ONE):
        if topics is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        a = tf.while_loop(
            c_criterion,
            lambda i, v, v_last, c, n, dist: (
                i + 1,
                tf.add(
                    tf.divide(i * v, i + int(math.sqrt(self.G.n))),
                    tf.scatter_nd(
                        tf.reshape(dist, shape=[int(math.sqrt(self.G.n)), 1]),
                        tf.fill([int(math.sqrt(self.G.n))],
                                1 / (int(math.sqrt(self.G.n)) + i)),
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
                tf.random_uniform([int(math.sqrt(self.G.n))], minval=0,
                                  maxval=self.G.n,
                                  dtype=tf.int32)
            ], name=self.name + "_while_conv")
        self.run(self.v.assign(a[1]))
        return self.v

    def _pr_steps_tf(self, steps, topics):
        if topics is not None:
            warnings.warn('Personalized PageRank not implemented yet!')

        a = tf.while_loop(
            lambda i, v, dist: i < steps,
            lambda i, v, dist: (
                i + 1.0,
                tf.add(
                    tf.divide(i * v, i + int(math.sqrt(self.G.n))),
                    tf.scatter_nd(
                        tf.reshape(dist, shape=[int(math.sqrt(self.G.n)), 1]),
                        tf.fill([int(math.sqrt(self.G.n))],
                                1 / (int(math.sqrt(self.G.n)) + i)),
                        [self.G.n])),
                tf.cast(tf.squeeze(tf.multinomial(
                    tf.nn.embedding_lookup(self.random_T.T_log, dist),
                    num_samples=1)), tf.int32)
            ),
            [
                0.0,
                self.v,
                tf.random_uniform([int(math.sqrt(self.G.n))], minval=0,
                                  maxval=self.G.n, dtype=tf.int32)
            ],
            name=self.name + "_while_steps")
        self.run(self.v.assign(a[1]))
        return self.v
