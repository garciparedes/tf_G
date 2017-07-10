import tensorflow as tf
import numpy as np
import timeit

import tf_G


def main():
  with tf.Session() as sess:
    g: tf_G.Graph = tf_G.GraphConstructor.unweighted_random(sess, "G", 10, 85)
    g_sparse: tf_G.Graph = tf_G.GraphConstructor.as_sparsifier(sess, g, 0.75)

    print(g)
    print(g.m)

    print(g_sparse)
    print(g_sparse.m)

    print(g_sparse.m / g.m)


if __name__ == '__main__':
  main()
