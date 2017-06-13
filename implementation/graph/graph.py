import tensorflow as tf

from tensor_flow_object import TensorFlowObject


class Graph(TensorFlowObject):
    def __init__(self, sess, name, edges_data_frame):
        TensorFlowObject.__init__(self, sess, name)
        self.edges_data_frame = edges_data_frame
        self.A_tf = tf.Variable(tf.transpose(
            tf.scatter_nd(self.E_list,
                          self.m * [1.0],
                          [self.n, self.n])), tf.float64, name=self.name + "_A")
        self.n_tf = tf.Variable(float(self.n), tf.float32,
                                name=self.name + "_n")

        self.sess.run(tf.variables_initializer([self.A_tf, self.n_tf]))

    @property
    def n(self):
        return int(self.edges_data_frame.max(axis=0).max() + 1)

    @property
    def m(self):
        return int(self.edges_data_frame.shape[0])

    @property
    def E_list(self):
        return self.edges_data_frame.values.tolist()

    @property
    def E_o_degrees(self):
        return tf.reduce_sum(self.A_tf, 0)
