import warnings
from typing import TypeVar

import tensorflow as tf

TF_type = TypeVar('TF_type', tf.Variable, tf.Tensor)


class TensorFlowObject(object):
  """ This class gives represents a TensorFlow object in the package.

  It acts as Parent class of many classes that uses the TensorFlow library and
  needs to execute code, so it's necessary to have a session and other
  attributes.

  """

  def __init__(self, sess: tf.Session, name: str,
               writer: tf.summary.FileWriter = None,
               is_sparse: bool = False) -> None:
    """ The constructor of class.

    It assign the input parameters to the class objects and no more.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      writer (:obj:`tf.summary.FileWriter`): This attribute represents a
        TensorFlow's Writer, that is used to obtain stats.
      is_sparse (bool): Use sparse Tensors if it's set to True. Not implemented
        yet. Show the Todo.

    Todo:
      * Implement variables as sparse when it's possible. Waiting to
        TensorFlow for it.

    """
    self.sess = sess
    self.name = name
    self.writer = writer
    if is_sparse:
      warnings.warn('TensorFlow not implements Sparse Variables yet!')
      # self.is_sparse = is_sparse

  def run_tf(self, input_to_run):
    """ Run method to execute TensorFlow operations

    Args:
      input_to_run: This parameter represents a TensorFlow operation.

    Returns:
      The result of the operation as numpy array

    """
    return self.sess.run(input_to_run)
