import tensorflow as tf
import numpy as np

import tf_G


def test_graph_cardinality():
  graph = tf_G.Graph(tf.Session(), "G_proof", edges_np=tf_G.DataSets.naive_6())

  assert graph.n == 6
  assert graph.m == 9


def test_graph_out_degrees():
  graph = tf_G.Graph(tf.Session(), "G_proof", edges_np=tf_G.DataSets.naive_6())

  np.testing.assert_array_equal(
    graph.out_degrees_np,
    np.array([[2.0], [2.0], [3.0], [1.0], [0.0], [1.0]]))


def test_graph_in_degrees():
  graph = tf_G.Graph(tf.Session(), "G_proof", edges_np=tf_G.DataSets.naive_6())

  np.testing.assert_array_equal(
    graph.in_degrees_np,
    np.array([[2.0, 1.0, 1.0, 2.0, 1.0, 2.0]]))


def test_graph_updateable():
  with tf.Session() as sess:
    g: tf_G.Graph = tf_G.GraphConstructor.from_edges(
      sess, "G1", edges_np=tf_G.DataSets.naive_6())

    g_updateable: tf_G.Graph = tf_G.GraphConstructor.empty(sess, "G2", g.n)

    for e in tf_G.DataSets.naive_6():
      g_updateable.append(e[0], e[1])

    np.testing.assert_array_equal(
      sess.run(g.A_tf),
      sess.run(g_updateable.A_tf))
