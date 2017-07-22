import tensorflow as tf
import numpy as np

import tfgraph


def test_algebraic_pagerank():
  with tf.Session() as sess:
    beta = 0.85
    graph = tfgraph.GraphConstructor.from_edges(sess, "G_proof",
                                             edges_np=tfgraph.DataSets.naive_6())
    pageRank = tfgraph.AlgebraicPageRank(sess, "Pr_Proof", graph, beta)

    np.testing.assert_array_almost_equal(
      pageRank.ranks_np(),
      np.array([[0.0, 0.321017],
                [5.0, 0.20074403],
                [1.0, 0.17054307],
                [3.0, 0.13679263],
                [2.0, 0.10659166],
                [4.0, 0.0643118]]),
      decimal=3
    )


def test_iterative_pagerank_upgradeable():
  beta = 0.85
  with tf.Session() as sess:
    g_upgradeable: tfgraph.Graph = tfgraph.GraphConstructor.empty(sess, "G", 6)
    pr_upgradeable: tfgraph.PageRank = tfgraph.AlgebraicPageRank(sess, "PR",
                                                          g_upgradeable, beta)

    for e in tfgraph.DataSets.naive_6():
      g_upgradeable.append(e[0], e[1])

    np.testing.assert_array_almost_equal(
      pr_upgradeable.ranks_np(),
      np.array([[0.0, 0.321017],
                [5.0, 0.20074403],
                [1.0, 0.17054307],
                [3.0, 0.13679263],
                [2.0, 0.10659166],
                [4.0, 0.0643118]]),
      decimal=2
    )
