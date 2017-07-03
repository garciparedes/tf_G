from enum import Enum
import tensorflow as tf


class VectorNorm(Enum):
  """ Enum class that contains some vectorial norms.

  The class has lambda functions that can be accessed as static class methods
  because it inherits Enum class.

  The methods operates with a vector with 1-D rank.

  """

  ONE = lambda x: tf.reduce_sum(tf.abs(x), 1)
  """ ONE norm.

  The one norm of the vector x.

  Returns:
    (:obj:`tf.Tensor`): Tensor that contains the one norm of the vector.       

  """

  EUCLIDEAN = lambda x: VectorNorm.P_NORM(x, 2)
  """ EUCLIDEAN norm.

  The euclidean norm of the vector x.

  Returns:
    (:obj:`tf.Tensor`): Tensor that contains the eucledian norm of the vector.       

  """

  P_NORM = lambda x, p: tf.pow(tf.reduce_sum(tf.pow(x, p)), 1 / p)
  """ P-NORM norm.

  The p-norm of the vector x.

  Returns:
    (:obj:`tf.Tensor`): Tensor that contains the p-norm of the vector.       

  """

  INFINITY = lambda x: tf.reduce_max(tf.abs(x), 1)
  """ INFINITY norm.

  The infinity norm of the vector x.

  Returns:
    (:obj:`tf.Tensor`): Tensor that contains the infinity norm of the vector.       

  """

  def __call__(self, *args, **kwargs):
    """ Call method of the class.

    It was overwritten because of by default the ENUM values not allows to call
    him.

    Args:
      *args: The args to the enum.
      **kwargs: The kwargs to the enum.

    Returns:
      The result of the corresponding enum's lambda function.

    """
    return self.value[0](*args, **kwargs)
