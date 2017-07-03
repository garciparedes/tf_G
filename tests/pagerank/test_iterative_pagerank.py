import tensorflow as tf
import numpy as np

import tf_G


def test_algebraic_pagerank():
  with tf.Session() as sess:
    beta = 0.85
    convergence = 0.01
    graph = tf_G.GraphConstructor.from_edges(sess, "G_proof",
                                             edges_np=tf_G.DataSets.naive_6())
    pageRank = tf_G.IterativePageRank(sess, "Pr_Proof", graph, beta)

    np.testing.assert_array_almost_equal(
      pageRank.ranks_np(convergence=convergence),
      np.array([[0.0, 0.321017],
                [5.0, 0.20074403],
                [1.0, 0.17054307],
                [3.0, 0.13679263],
                [2.0, 0.10659166],
                [4.0, 0.0643118]]),
      decimal=1
    )
