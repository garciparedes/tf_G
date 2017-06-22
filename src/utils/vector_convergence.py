from enum import Enum
import tensorflow as tf

from src.utils.vector_norm import VectorNorm


class VectorConvergenceCriterion(Enum):
    ONE = lambda x, y, c, n: VectorNorm.ONE(
        tf.subtract(x, y)) > [c]

    INFINITY = lambda x, y, c, n: VectorNorm.INFINITY(
        tf.subtract(x, y)) > [c / n]

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)
