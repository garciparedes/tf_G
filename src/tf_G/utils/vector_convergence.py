from enum import Enum
import tensorflow as tf

from tf_G.utils.vector_norm import VectorNorm


class ConvergenceCriterion(Enum):
    ONE = lambda i, x, y, c, n, dist=None: tf.reshape(
        VectorNorm.ONE(tf.subtract(x, y)) > [c], [])

    INFINITY = lambda i, x, y, c, n,dist=None: tf.reshape(VectorNorm.INFINITY(
        tf.subtract(x, y)) > [c / n], [])

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)
