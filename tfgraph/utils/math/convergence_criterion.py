from enum import Enum

import tensorflow as tf

from tfgraph.utils.math.vector_norm import VectorNorm


class ConvergenceCriterion(Enum):
  """ Enum class that contains some convergence criteria.

  The class has lambda functions that can be accessed as static class methods
  because it inherits Enum class.

  The methods operates with two vectors and a convergence bound value.

  """

  ONE = lambda i=None, x=None, y=None, c=None, n=None, dist=None: tf.reshape(
    VectorNorm.ONE(tf.subtract(x, y)) > [c], [])
  """ ONE convergence criterion.
  
  The one convergence criterion uses the norm one to gets the difference between 
  two vectors and the compare it with the convergence bound c.
  
  Returns:
    (:obj:`tf.Tensor`): Tensor that returns True if the difference of the 
      vectors is inside the convergence criterion interval, and False otherwise.        
  
  """

  INFINITY = lambda i=None, x=None, y=None, c=None, n=None, dist=None: \
    tf.reshape(VectorNorm.INFINITY(
    tf.subtract(x, y)) > [c / n], [])
  """ INFINITY convergence criterion.

  The one convergence criterion uses the norm infinity to gets the difference 
  between two vectors and the compare it with the convergence bound c.

  Returns:
    (:obj:`tf.Tensor`): Tensor that returns True if the difference of the 
      vectors is inside the convergence criterion interval, and False otherwise.        
  
  """

  def __call__(self, *args, **kwargs):
    """ Call method of the class.

    It was overwriten because of by default the ENUM values not allows to call
    him.

    Args:
      *args: The args to the enum.
      **kwargs: The kwargs to the enum.

    Returns:
      The result of the corresponding enum's lambda function.

    """
    return self.value[0](*args, **kwargs)
