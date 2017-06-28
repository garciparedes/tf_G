import math
import warnings
from typing import List

import tensorflow as tf

from tfg_big_data_algorithms.pagerank.transition.transition_random import \
    TransitionRandom
from tfg_big_data_algorithms.pagerank.numeric_iterative_pagerank import \
    NumericIterativePageRank
from tfg_big_data_algorithms.utils.vector_convergence import \
    ConvergenceCriterion
from tfg_big_data_algorithms.graph.graph import Graph


class NumericRandomWalkPageRank(NumericIterativePageRank):
    def __init__(self, sess: tf.Session, name: str, graph: Graph,
                 beta: float) -> None:
        warnings.warn('This implementation is in alpha. ' +
                      'Use NumericIterativePageRank Implementation!')

        NumericIterativePageRank.__init__(self, sess, name + "_rw", graph, beta)

        self.random_T = TransitionRandom(sess, self.name, self.G, self.beta)

        self.iter = lambda t, v, n=self.G.n: tf.add(
            tf.divide((v * t), tf.add(t, n)),
            self.random_T(t))

    def _pr_convergence_tf(self, convergence: float, topics: List[int] = None,
                           c_criterion=ConvergenceCriterion.ONE) -> tf.Tensor:
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
                    tf.nn.embedding_lookup(self.random_T.transition, dist),
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

    def _pr_steps_tf(self, steps: int, topics: List[int] = None) -> tf.Tensor:
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
                    tf.nn.embedding_lookup(self.random_T.transition, dist),
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
