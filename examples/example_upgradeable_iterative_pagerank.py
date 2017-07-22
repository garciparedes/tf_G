#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import timeit

import tfgraph


def main():
  beta: float = 0.85
  convergence: float = 0.01

  edges_np: np.ndarray = tfgraph.DataSets.naive_6()

  with tf.Session() as sess:
    writer: tf.summary.FileWriter = tf.summary.FileWriter(
      'logs/tensorflow/.')

    graph: tfgraph.Graph = tfgraph.GraphConstructor.from_edges(
      sess, "G1", edges_np, writer, is_sparse=False)

    pr_iter: tfgraph.PageRank = tfgraph.IterativePageRank(
      sess, "PR1", graph, beta)

    g_upgradeable: tfgraph.Graph = tfgraph.GraphConstructor.empty(
      sess, "G2", graph.n, writer)

    pr_upgradeable: tfgraph.PageRank = tfgraph.IterativePageRank(
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
