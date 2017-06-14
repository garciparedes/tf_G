import tensorflow as tf

from tensor_flow_object import TensorFlowObject


class Graph(TensorFlowObject):
    def __init__(self, sess, name, edges_data_frame):
        TensorFlowObject.__init__(self, sess, name)
        self.edges_df = edges_data_frame

        if self.n != 0:
            A_init = tf.scatter_nd(self.E_list, self.m * [1.0],
                                   [self.n, self.n])
        else:
            A_init = tf.zeros([1, 1])
        self.A_tf = tf.Variable(A_init, tf.float64, name=self.name + "_A")
        self.n_tf = tf.Variable(float(self.n), tf.float32,
                                name=self.name + "_n")
        self.sess.run(tf.variables_initializer([self.A_tf, self.n_tf]))

    @property
    def n(self):
        try:
            return int(self.edges_df.max(axis=0).max() + 1)
        except ValueError:
            return 0

    @property
    def m(self):
        return int(self.edges_df.shape[0])

    @property
    def E_list(self):
        return self.edges_df.values.tolist()

    @property
    def is_not_sink_vertice(self):
        return tf.not_equal(tf.reduce_sum(self.A_tf, 1), 0)

    @property
    def E_o_degrees(self, keep_dims=True):
        return tf.reduce_sum(self.A_tf, 1, keep_dims=keep_dims)

    def __str__(self):
        return str(self.edges_df)