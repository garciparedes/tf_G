#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import timeit

import tfgraph


def main():
  beta: float = 0.85
  convergence: float = 0.0001

  edges_np: np.ndarray = tfgraph.DataSets.naive_6()

  with tf.Session() as sess:
    writer: tf.summary.FileWriter = tf.summary.FileWriter('logs/tensorflow/')

    graph: tfgraph.Graph = tfgraph.GraphConstructor.from_edges(
      sess, "G", edges_np, is_sparse=False)

    pr_itr: tfgraph.PageRank = tfgraph.IterativePageRank(sess, "PR", graph, beta)

    start_time: float = timeit.default_timer()
    b: np.ndarray = pr_itr.ranks_np(convergence=convergence)
    elapsed: float = timeit.default_timer() - start_time
    print(elapsed)

    print(b)
    writer.add_graph(sess.graph)


if __name__ == '__main__':
  main()
