import tensorflow as tf
import numpy as np

from tf_G.utils.tensorflow_object import TensorFlowObject, \
    TF_type
from tf_G.utils.update_edge_notifier import \
    UpdateEdgeNotifier


class Graph(TensorFlowObject, UpdateEdgeNotifier):
    def __init__(self, sess: tf.Session, name: str,
                 writer: tf.summary.FileWriter = None,
                 edges_np: np.ndarray = None, n: int = None,
                 is_sparse: bool = False) -> None:
        TensorFlowObject.__init__(self, sess, name, writer, is_sparse)
        UpdateEdgeNotifier.__init__(self)

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

        self.n_tf: TF_type = tf.Variable(float(self.n), tf.float32,
                                         name=self.name + "_n")
        self.A_tf: TF_type = tf.Variable(A_init, tf.float64,
                                         name=self.name + "_A")
        self.out_degrees_tf: TF_type = tf.Variable(
            tf.reduce_sum(self.A_tf, 1, keep_dims=True),
            name=self.name + "_d_out")
        self.in_degrees_tf: TF_type = tf.Variable(
            tf.reduce_sum(self.A_tf, 0, keep_dims=True),
            name=self.name + "_d_in")
        self.L_tf: TF_type = tf.Variable(
            tf.diag(self.get_out_degrees_tf()) - self.A_tf,
            name=self.name + "_L")
        self.run(tf.variables_initializer([self.A_tf, self.n_tf]))
        self.run(tf.variables_initializer([
            self.out_degrees_tf, self.in_degrees_tf]))
        self.run(tf.variables_initializer([self.L_tf]))

    @property
    def is_not_sink_tf(self) -> tf.Tensor:
        return tf.not_equal(self.get_out_degrees_tf(), 0)

    @property
    def out_degrees_np(self) -> np.ndarray:
        return self.run(self.get_out_degrees_tf())

    @property
    def out_degrees_np(self) -> np.ndarray:
        return self.run(self.out_degrees_tf)

    @property
    def edge_list_np(self) -> np.ndarray:
        return self.run(self.edge_list_tf)

    @property
    def edge_list_tf(self) -> tf.Tensor:
        return tf.cast(tf.where(tf.not_equal(self.A_tf, 0)), tf.int64)

    @property
    def L_pseudo_inverse_tf(self) -> tf.Tensor:
        return tf.py_func(np.linalg.pinv, [self.L_tf], tf.float32)

    def A_tf_vertex(self, vertex: int) -> tf.Tensor:
        return tf.gather(self.A_tf, [vertex])

    def out_degrees_tf_vertex(self, vertex: int) -> tf.Tensor:
        return tf.gather(self.out_degrees_tf, [vertex])

    def is_not_sink_tf_vertex(self, vertex: int) -> TF_type:
        return tf.not_equal(
            tf.reshape([self.out_degrees_tf_vertex(vertex)], [1]), 0)

    def get_in_degrees_tf(self, keep_dims=False) -> TF_type:
        if keep_dims is False:
            return tf.reshape(self.in_degrees_tf, [self.n])
        else:
            return self.in_degrees_tf

    def get_out_degrees_tf(self, keep_dims=False) -> TF_type:
        if keep_dims is False:
            return tf.reshape(self.out_degrees_tf, [self.n])
        else:
            return self.out_degrees_tf

    def __str__(self) -> str:
        return str(self.run(self.L_tf))

    def append(self, src: int, dst: int) -> None:
        if src and dst is None:
            raise ValueError(
                "tf_G and dst must not be None ")
        self.run([tf.scatter_nd_add(self.A_tf, [[src, dst]], [1.0]),
                  tf.scatter_nd_add(self.out_degrees_tf, [[src, 0]], [1.0]),
                  tf.scatter_nd_add(self.in_degrees_tf, [[0, dst]], [1.0]),
                  tf.scatter_nd_add(self.L_tf, [[src, src], [src, dst]],
                                    [+1.0, -1.0])])
        self.m += 1
        self._notify([src, dst], 1)

    def remove(self, src: int, dst: int) -> None:
        if src and dst is None:
            raise ValueError(
                "tf_G and dst must not be None ")
        self.run([tf.scatter_nd_add(self.A_tf, [[src, dst]], [-1.0]),
                  tf.scatter_nd_add(self.out_degrees_tf, [[src, 0]], [-1.0]),
                  tf.scatter_nd_add(self.in_degrees_tf, [[0, dst]], [-1.0]),
                  tf.scatter_nd_add(self.L_tf, [[src, src], [src, dst]],
                                    [-1.0, +1.0])])
        self.m -= 1
        self._notify([src, dst], -1)
