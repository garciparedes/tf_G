import tensorflow as tf

from pagerank.numeric_page_rank import NumericPageRank


class NumericIterativePageRank(NumericPageRank):
    def __init__(self, sess, name, graph, reset_probability=None):
        NumericPageRank.__init__(self, sess, name, graph, reset_probability)
        self.v_last = tf.Variable(tf.fill([self.G.n, 1], 0.0),
                                  name=self.name + "_Vi-1")
        self.iteration = tf.assign(self.v,
                                   tf.matmul(self.transition.get, self.v,
                                             a_is_sparse=True))
        self.sess.run(tf.variables_initializer([self.v_last]))

    def page_rank_until_convergence(self, convergence):
        diff = tf.reduce_max(tf.abs(tf.subtract(self.v_last, self.v)), 0)
        self.sess.run(tf.assign(self.v_last, self.v))
        self.sess.run(self.iteration)
        while self.sess.run(diff)[0] > convergence / self.sess.run(
                self.G.n_tf):
            self.sess.run(tf.assign(self.v_last, self.v))
            self.sess.run(self.iteration)
        return self.sess.run(self.v)

    def page_rank_until_steps(self, steps):
        for step in range(steps):
            self.sess.run(self.iteration)
        return self.sess.run(self.v)
