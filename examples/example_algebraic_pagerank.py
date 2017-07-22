#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import timeit

import tfgraph


def main():
  beta: float = 0.85

  edges_np: np.ndarray = tfgraph.DataSets.naive_6()

  with tf.Session() as sess:
    writer: tf.summary.FileWriter = tf.summary.FileWriter('logs/tensorflow/')

    graph: tfgraph.Graph = tfgraph.GraphConstructor.from_edges(
      sess, "G", edges_np, is_sparse=False)

    pr_alg: tfgraph.PageRank = tfgraph.AlgebraicPageRank(sess, "PR", graph, beta)

    start_time: float = timeit.default_timer()
    a: np.ndarray = pr_alg.ranks_np()
    elapsed: float = timeit.default_timer() - start_time
    print(elapsed)

    print(a)

    writer.add_graph(sess.graph)


if __name__ == '__main__':
  main()
