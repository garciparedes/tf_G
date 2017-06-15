import tensorflow as tf

from tensor_flow_object import TensorFlowObject


class Graph(TensorFlowObject):
    def __init__(self, sess, name, edges_np=None, n=None):
        TensorFlowObject.__init__(self, sess, name)

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
        self.sess.run(tf.variables_initializer([self.A_tf, self.n_tf]))

    @property
    def is_not_sink_vertice(self):
        return tf.not_equal(tf.reduce_sum(self.A_tf, 1), 0)

    @property
    def E_o_degrees(self, keep_dims=True):
        return tf.reduce_sum(self.A_tf, 1, keep_dims=keep_dims)

    def __str__(self):
        return str(self.sess.run(self.A_tf))

    def append(self, src, dst):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.sess.run(tf.scatter_nd_update(self.A_tf, [[src, dst]], [1.0]))
        self.m += 1

    def remove(self, src=None, dst=None):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.sess.run(tf.scatter_nd_update(self.A_tf, [[src, dst]], [-1.0]))

        self.m -= 1
