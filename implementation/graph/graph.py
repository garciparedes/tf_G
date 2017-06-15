import tensorflow as tf

from tensor_flow_object import TensorFlowObject


class Graph(TensorFlowObject):
    def __init__(self, sess, name, writer=None, edges_np=None, n=None):
        TensorFlowObject.__init__(self, sess, name, writer)

        if edges_np is not None:
            self.n = int(edges_np.max(axis=0).max() + 1)
            self.m = int(edges_np.shape[0])
            A_init = tf.scatter_nd(edges_np.tolist(), self.m * [1.0],
                                   [self.n, self.n])
        elif n is not None:
            self.n = n
            self.m = 0
            A_init = tf.zeros([self.n, self.n])
        else:
            raise ValueError('Graph constructor must be have edges or n')

        self.A_tf = tf.Variable(A_init, tf.float64,
                                name=self.name + "_A")
        self.n_tf = tf.Variable(float(self.n), tf.float32,
                                name=self.name + "_n")
        self.run(tf.variables_initializer([self.A_tf, self.n_tf]))
        self.generate_summaries()

    def generate_summaries(self):
        self.writer.add_summary(
            self.run(tf.summary.scalar(self.name + '_vertices', self.n)))
        self.writer.add_summary(
            self.run(tf.summary.scalar(self.name + '_edges', self.m)))
        self.writer.add_summary(
            self.run(
                tf.summary.histogram(self.name + '_in-degrees',
                                     tf.concat(([self.get_in_degrees()],
                                                [tf.range(0, self.n,
                                                          dtype=tf.float32)]),
                                               0))))
        self.writer.add_summary(
            self.run(
                tf.summary.histogram(self.name + '_out-degrees',
                                     tf.concat(([self.get_out_degrees()],
                                                [tf.range(0, self.n,
                                                          dtype=tf.float32)]),
                                               0))))

    @property
    def is_not_sink_vertice(self):
        return tf.not_equal(self.get_out_degrees(), 0)

    @property
    def in_degrees(self):
        return self.get_in_degrees(keep_dims=True)

    @property
    def out_degrees(self):
        return self.get_out_degrees(keep_dims=True)

    def get_in_degrees(self, keep_dims=False):
        return tf.reduce_sum(self.A_tf, 0, keep_dims=keep_dims)

    def get_out_degrees(self, keep_dims=False):
        return tf.reduce_sum(self.A_tf, 1, keep_dims=keep_dims)

    def __str__(self):
        return str(self.run(self.A_tf))

    def append(self, src, dst):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.run(tf.scatter_nd_update(self.A_tf, [[src, dst]], [1.0]))
        self.m += 1

    def remove(self, src=None, dst=None):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.run(tf.scatter_nd_update(self.A_tf, [[src, dst]], [-1.0]))

        self.m -= 1
