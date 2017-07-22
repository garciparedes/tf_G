#!/usr/bin/python3

import tensorflow as tf
import numpy as np

import tfgraph


def main():
  beta: float = 0.85
  convergence: float = 0.0001

  edges_np: np.ndarray = tfgraph.DataSets.naive_6()

  with tf.Session() as sess:
    graph: tfgraph.Graph = tfgraph.GraphConstructor.from_edges(
      sess, "G", edges_np, is_sparse=False)

    pr_iter: tfgraph.PageRank = tfgraph.IterativePageRank(sess, "PR1", graph, beta)

    v_4: np.ndarray = pr_iter.pagerank_vector_np(convergence=convergence,
                                                 topics=[4])
    v_5: np.ndarray = pr_iter.pagerank_vector_np(convergence=convergence,
                                                 topics=[5])

    v_45: np.ndarray = pr_iter.pagerank_vector_np(convergence=convergence,
                                                  topics=[4, 5])
    v_45_pseudo: np.ndarray = (v_4 + v_5) / 2.0

    print(v_4)
    print(np.sum(v_4))
    print(v_5)
    print(np.sum(v_5))
    print(v_45)
    print(np.sum(v_45))
    print(v_45_pseudo)
    print(np.sum(v_45_pseudo))


if __name__ == '__main__':
  main()
