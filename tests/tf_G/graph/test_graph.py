import tensorflow as tf
import numpy as np

import tfgraph


def test_graph_cardinality():
  graph = tfgraph.Graph(tf.Session(), "G_proof", edges_np=tfgraph.DataSets.naive_6())

  assert graph.n == 6
  assert graph.m == 9


def test_graph_out_degrees():
  graph = tfgraph.Graph(tf.Session(), "G_proof", edges_np=tfgraph.DataSets.naive_6())

  np.testing.assert_array_equal(
    graph.out_degrees_np,
    np.array([[2.0], [2.0], [3.0], [1.0], [0.0], [1.0]]))


def test_graph_in_degrees():
  graph = tfgraph.Graph(tf.Session(), "G_proof", edges_np=tfgraph.DataSets.naive_6())

  np.testing.assert_array_equal(
    graph.in_degrees_np,
    np.array([[2.0, 1.0, 1.0, 2.0, 1.0, 2.0]]))


def test_graph_upgradeable():
  with tf.Session() as sess:
    g: tfgraph.Graph = tfgraph.GraphConstructor.from_edges(
      sess, "G1", edges_np=tfgraph.DataSets.naive_6())

    g_upgradeable: tfgraph.Graph = tfgraph.GraphConstructor.empty(sess, "G2", g.n)

    for e in tfgraph.DataSets.naive_6():
      g_upgradeable.append(e[0], e[1])

    np.testing.assert_array_equal(
      sess.run(g.A_tf),
      sess.run(g_upgradeable.A_tf))
