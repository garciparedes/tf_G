import tensorflow as tf

from pagerank.page_rank import PageRank
from pagerank.transition_matrix import TransitionMatrix
from pagerank.transition_reset_matrix import TransitionResetMatrix
from utils import Utils
import numpy as np


class NumericPageRank(PageRank):
    def __init__(self, sess, name, graph, reset_probability=None):
        PageRank.__init__(self, sess, name)
        self.G = graph
        if reset_probability is None:
            self.transition = TransitionMatrix(self.sess, self.name, self.G)
        else:
            self.transition = TransitionResetMatrix(self.sess, self.name, self.G,
                                                    reset_probability)
        self.v = tf.Variable(tf.fill([self.G.n, 1], tf.pow(self.G.n_tf, -1)),
                             name=self.name + "_Vi")
        self.page_rank = None
        self.sess.run(tf.variables_initializer([self.v]))

    def ranks(self, convergence=None, steps=None):
        if convergence or steps is not None:
            self.page_rank_vector(convergence, steps)
        ranks = tf.py_func(Utils.ranked, [tf.multiply(self.v, -1)], tf.int64)
        ranks = tf.map_fn(lambda x: [x, tf.gather(self.v, x)[0]], ranks,
                          dtype=[tf.int64, tf.float32])
        tf.summary.FileWriter('logs/.', self.sess.graph)
        return np.concatenate(self.sess.run(ranks), axis=1)
