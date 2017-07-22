import tensorflow as tf
import numpy as np

import tfgraph


def test_convergence_criterion_one():
  with tf.Session() as sess:
    assert False == sess.run(tfgraph.ConvergenceCriterion.INFINITY(
      x=tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]),
      y=tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]),
      c=0.1,
      n=5
    ))

    assert True == sess.run(tfgraph.ConvergenceCriterion.INFINITY(
      x=tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]),
      y=tf.constant([[3.0, 2.0, -3.0, 4.0, 5.0]]),
      c=0.1,
      n=5
    ))


def test_convergence_criterion_infinity():
  with tf.Session() as sess:
    assert False == sess.run(tfgraph.ConvergenceCriterion.ONE(
      x=tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]),
      y=tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]),
      c=0.1
    ))

    assert True == sess.run(tfgraph.ConvergenceCriterion.ONE(
      x=tf.constant([[1.0, 2.0, -3.0, 4.0, 5.0]]),
      y=tf.constant([[3.0, 2.0, -3.0, 4.0, 5.0]]),
      c=0.1
    ))
