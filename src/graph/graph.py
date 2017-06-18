import tensorflow as tf
import numpy as np

from src.utils.notifier import Notifier
from src.utils.tensorflow_object import TensorFlowObject


class Graph(TensorFlowObject, Notifier):
    def __init__(self, sess, name, writer=None, edges_np=None, n=None):
        TensorFlowObject.__init__(self, sess, name, writer)
        Notifier.__init__(self)

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

        self.n_tf = tf.Variable(float(self.n), tf.float32,
                                name=self.name + "_n")
        self.A_tf = tf.Variable(A_init, tf.float64,
                                name=self.name + "_A")
        self.L_tf = tf.Variable(tf.diag(self.get_out_degrees()) - self.A_tf,
                                name=self.name + "_L")

        self.run(tf.variables_initializer([self.A_tf, self.n_tf]))
        self.run(tf.variables_initializer([self.L_tf]))

    @property
    def is_not_sink_vertice(self):
        return tf.not_equal(self.get_out_degrees(), 0)

    @property
    def in_degrees(self):
        return self.get_in_degrees(keep_dims=True)

    @property
    def out_degrees(self):
        return self.get_out_degrees(keep_dims=True)

    @property
    def edges_np(self):
        return None

    @property
    def L_pseudo_inverse(self):
        return tf.py_func(np.linalg.pinv, [self.L_tf], tf.float32)

    def get_in_degrees(self, keep_dims=False):
        return tf.reduce_sum(self.A_tf, 0, keep_dims=keep_dims)

    def get_out_degrees(self, keep_dims=False):
        return tf.reduce_sum(self.A_tf, 1, keep_dims=keep_dims)

    def __str__(self):
        return str(self.run(self.L_pseudo_inverse))

    def append(self, src, dst):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.run(tf.scatter_nd_update(self.A_tf, [[src, dst]], [1.0]))
        self.m += 1
        self._notify()

    def remove(self, src, dst):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.run(tf.scatter_nd_update(self.A_tf, [[src, dst]], [-1.0]))
        self.m -= 1
        self._notify()

    def sparsifier(self, alpha):
        '''
        boolean_distribution = tf.less_equal(
            tf.random_uniform([self.m], 0.0, 1.0), alpha)
        edges_np = self.edges_np[self.sess.run(boolean_distribution)]
        '''
        return Graph(self.sess, self.name + "_sparsifier",
                     edges_np=self.edges_np)
