#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import timeit

import tf_G


def main():
  beta: float = 0.85
  convergence: float = 0.01

  edges_np: np.ndarray = tf_G.DataSets.naive_6()

  with tf.Session() as sess:
    writer: tf.summary.FileWriter = tf.summary.FileWriter(
      'logs/tensorflow/.')

    graph: tf_G.Graph = tf_G.GraphConstructor.from_edges(
      sess, "G1", edges_np, writer, is_sparse=False)

    pr_iter: tf_G.PageRank = tf_G.IterativePageRank(
      sess, "PR1", graph, beta)

    g_upgradeable: tf_G.Graph = tf_G.GraphConstructor.empty(
      sess, "G2", graph.n, writer)

    pr_upgradeable: tf_G.PageRank = tf_G.IterativePageRank(
      sess, "PR2", g_upgradeable, beta)

    b: np.ndarray = pr_iter.ranks_np(convergence=convergence)


    print(b)

    for e in edges_np:
      g_upgradeable.append(e[0], e[1])
      print("[" + str(e[0]) + ", " + str(e[1]) + "]")

    e = pr_upgradeable.ranks_np(convergence=convergence)
    print(e)

    writer.add_graph(sess.graph)


if __name__ == '__main__':
  main()
