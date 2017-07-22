#!/usr/bin/python3

import tensorflow as tf
import tfgraph


def main():
  with tf.Session() as sess:
    g: tfgraph.Graph = tfgraph.GraphConstructor.unweighted_random(sess, "G", 10, 85)
    g_sparse: tfgraph.Graph = tfgraph.GraphConstructor.as_sparsifier(sess, g, 0.75)

    print(g)
    print(g.m)

    print(g_sparse)
    print(g_sparse.m)

    print(g_sparse.m / g.m)


if __name__ == '__main__':
  main()
