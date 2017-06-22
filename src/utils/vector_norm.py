from enum import Enum
import tensorflow as tf


class VectorNorm(Enum):
    ONE = lambda x: tf.reduce_sum(tf.abs(x), 1)
    TWO = lambda x, y: tf.reduce_sum(tf.multiply(x, y), 1)
    EUCLIDEAN = lambda x: VectorNorm.P_NORM(x,2)
    P_NORM = lambda x, p: tf.pow(tf.reduce_sum(tf.pow(x, p)), -p)
    INFINITY = lambda x: tf.reduce_max(tf.abs(x), 1)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)