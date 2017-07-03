import tensorflow as tf

import tf_G


def test_graph_constructor_empty():
  n = 10
  graph = tf_G.GraphConstructor.empty(tf.Session(), "G_proof", n=n)

  assert graph.n == n
  assert graph.m == 0


def test_graph_constructor_unweighted_random():
  n = 100
  m = 1000
  graph = tf_G.GraphConstructor.unweighted_random(tf.Session(), "G_proof", n=n,
                                                  m=m)

  assert graph.n == n
  assert graph.m == m
  
