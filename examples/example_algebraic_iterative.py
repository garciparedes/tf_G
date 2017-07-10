import tensorflow as tf
import numpy as np
import timeit

import tf_G


def main():
  beta: float = 0.85
  convergence: float = 0.0001

  edges_np: np.ndarray = tf_G.DataSets.followers()

  with tf.Session() as sess:
    graph: tf_G.Graph = tf_G.GraphConstructor.from_edges(
      sess, "G", edges_np, is_sparse=False)

    pr_alg: tf_G.PageRank = tf_G.AlgebraicPageRank(sess, "PR_alg", graph, beta)

    pr_itr: tf_G.PageRank = tf_G.IterativePageRank(sess, "PR_itr", graph, beta)

    start_time: float = timeit.default_timer()
    a: np.ndarray = pr_alg.ranks_np()
    elapsed: float = timeit.default_timer() - start_time
    print(elapsed)

    start_time: float = timeit.default_timer()
    b: np.ndarray = pr_itr.ranks_np(convergence=convergence)
    elapsed: float = timeit.default_timer() - start_time
    print(elapsed)

    print(a)
    print(b)


if __name__ == '__main__':
  main()
