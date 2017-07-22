import tensorflow as tf
import numpy as np

import tfgraph


def test_iterative_pagerank_convergence():
  with tf.Session() as sess:
    beta = 0.85
    convergence = 0.01
    graph = tfgraph.GraphConstructor.from_edges(sess, "G_proof",
                                             edges_np=tfgraph.DataSets.naive_6())
    pageRank = tfgraph.IterativePageRank(sess, "Pr_Proof", graph, beta)

    np.testing.assert_array_almost_equal(
      pageRank.ranks_np(convergence=convergence),
      np.array([[0.0, 0.321017],
                [5.0, 0.20074403],
                [1.0, 0.17054307],
                [3.0, 0.13679263],
                [2.0, 0.10659166],
                [4.0, 0.0643118]]),
      decimal=2
    )


def test_iterative_pagerank_steps():
  with tf.Session() as sess:
    beta = 0.85
    steps = 100
    graph = tfgraph.GraphConstructor.from_edges(sess, "G_proof",
                                             edges_np=tfgraph.DataSets.naive_6())
    pageRank = tfgraph.IterativePageRank(sess, "Pr_Proof", graph, beta)

    np.testing.assert_array_almost_equal(
      pageRank.ranks_np(steps=steps),
      np.array([[0.0, 0.321017],
                [5.0, 0.20074403],
                [1.0, 0.17054307],
                [3.0, 0.13679263],
                [2.0, 0.10659166],
                [4.0, 0.0643118]]),
      decimal=2
    )


def test_iterative_personalized_pagerank_convergence():
  with tf.Session() as sess:
    beta = 0.85
    convergence = 0.001
    graph = tfgraph.GraphConstructor.from_edges(sess, "G_proof",
                                             edges_np=tfgraph.DataSets.naive_6())
    pageRank = tfgraph.IterativePageRank(sess, "Pr_Proof", graph, beta)

    print(pageRank.ranks_np(convergence=convergence, topics=[4]))
    np.testing.assert_array_almost_equal(
      pageRank.ranks_np(convergence=convergence, topics=[4]),
      np.array([[4, 0.999186],
                [0, 0.000284],
                [5, 0.000176],
                [1, 0.000150],
                [3, 0.000116],
                [2, 0.000089]]),
      decimal=2
    )


def test_iterative_personalized_pagerank_steps():
  with tf.Session() as sess:
    beta = 0.85
    steps = 100
    graph = tfgraph.GraphConstructor.from_edges(sess, "G_proof",
                                             edges_np=tfgraph.DataSets.naive_6())
    pageRank = tfgraph.IterativePageRank(sess, "Pr_Proof", graph, beta)

    np.testing.assert_array_almost_equal(
      pageRank.ranks_np(steps=steps, topics=[4]),
      np.array([[4, 0.999186],
                [0, 0.000284],
                [5, 0.000176],
                [1, 0.000150],
                [3, 0.000116],
                [2, 0.000089]]),
      decimal=2
    )


def test_iterative_pagerank_upgradeable():
  beta = 0.85
  convergence=0.01
  with tf.Session() as sess:

    g_upgradeable: tfgraph.Graph = tfgraph.GraphConstructor.empty(sess, "G", 6)
    pr_upgradeable: tfgraph.PageRank = tfgraph.IterativePageRank(
      sess, "PR", g_upgradeable, beta)

    for e in tfgraph.DataSets.naive_6():
      g_upgradeable.append(e[0], e[1])

    np.testing.assert_array_almost_equal(
      pr_upgradeable.ranks_np(convergence=convergence),
      np.array([[0.0, 0.321017],
                [5.0, 0.20074403],
                [1.0, 0.17054307],
                [3.0, 0.13679263],
                [2.0, 0.10659166],
                [4.0, 0.0643118]]),
      decimal=2
    )
