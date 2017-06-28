import warnings
from typing import TypeVar

import tensorflow as tf


TF_type = TypeVar('TF_type', tf.Variable, tf.Tensor)


class TensorFlowObject(object):
    def __init__(self, sess: tf.Session, name: str,
                 writer: tf.summary.FileWriter = None,
                 is_sparse: bool = False) -> None:
        self.sess = sess
        self.name = name
        self.writer = writer
        if is_sparse:
            warnings.warn('TensorFlow not implements Sparse Variables yet!')
            # self.is_sparse = is_sparse

    def run(self, input_to_run):
        return self.sess.run(input_to_run)
