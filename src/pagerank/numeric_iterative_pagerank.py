import tensorflow as tf

from src.pagerank.numeric_pagerank import NumericPageRank
from src.pagerank.transition_reset_matrix import TransitionResetMatrix


class NumericIterativePageRank(NumericPageRank):
    def __init__(self, sess, name, graph, beta=None):
        T = TransitionResetMatrix(sess, name, graph, beta)
        NumericPageRank.__init__(self, sess, name, graph, beta, T)
        self.v_last = tf.Variable(tf.zeros([1, self.G.n]),
                                  name=self.name + "_Vi-1")
        self.iter = [self.v_last.assign(self.v),
                     self.v.assign(tf.matmul(self.v, self.T.get,
                                             b_is_sparse=True))]
        self.run(tf.variables_initializer([self.v_last]))

    def _page_rank_until_convergence(self, convergence, personalized=None):
        if personalized is not None:
            pass
        else:
            pass

        diff = tf.gather(
            tf.reduce_max(tf.abs(tf.subtract(self.v_last, self.v)), 1), 0)
        self.run(self.iter)
        while self.run(diff > convergence / self.G.n_tf):
            self.run(self.iter)
        return self.run(self.v)

    def _page_rank_until_steps(self, steps, personalized=None):
        if personalized:
            pass
        else:
            pass
        for step in range(steps):
            self.run(self.iter)
        return self.run(self.v)

    def _page_rank_exact(self, personalized=None):
        raise NotImplementedError(
            'NumericIterativePageRank not implements exact PageRank')
