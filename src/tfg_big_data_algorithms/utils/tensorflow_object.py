import warnings
from typing import TypeVar

import tensorflow as tf


TF_type = TypeVar('TF_type', tf.Variable, tf.Tensor)


class TensorFlowObject(object):
    """
    This is a proof of class documentation

    """
    def __init__(self, sess: tf.Session, name: str,
                 writer: tf.summary.FileWriter = None,
                 is_sparse: bool = False) -> None:
        """
        This is a proof of __init__ documentation
        
        :param sess:
        :param name:
        :param writer:
        :param is_sparse:
        """
        self.sess = sess
        self.name = name
        self.writer = writer
        if is_sparse:
            warnings.warn('TensorFlow not implements Sparse Variables yet!')
            # self.is_sparse = is_sparse

    def run(self, input_to_run):
        """
        This is a proof of method documentation

        :param input_to_run:
        :return:
        """
        return self.sess.run(input_to_run)
