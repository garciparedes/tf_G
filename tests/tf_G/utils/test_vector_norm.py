import tensorflow as tf
import numpy as np
import tf_G


def test_one():
  with tf.Session() as sess:
    np.testing.assert_array_equal(
      sess.run(tf_G.VectorNorm.ONE(tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]))),
      np.array([15.0]))


def test_euclidean():
  with tf.Session() as sess:
    np.testing.assert_array_almost_equal(
      sess.run(
        tf_G.VectorNorm.EUCLIDEAN(tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]]))),
      np.array([7.41]), decimal=2)


def test_p_norm():
  with tf.Session() as sess:
    np.testing.assert_array_almost_equal(
      sess.run(
        tf_G.VectorNorm.P_NORM(tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]]), 3)),
      np.array([6.08]), decimal=2)


def test_infinity():
  with tf.Session() as sess:
    np.testing.assert_array_equal(
      sess.run(
        tf_G.VectorNorm.INFINITY(tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]]))),
      np.array([5.0]))
