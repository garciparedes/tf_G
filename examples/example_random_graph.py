#!/usr/bin/python3

import tensorflow as tf

import tfgraph


def main():
  with tf.Session() as sess:

    g_1 = tfgraph.GraphConstructor.unweighted_random(sess, "G_1_rand", 10, 15)
    g_2 = tfgraph.GraphConstructor.unweighted_random(sess, "G_2_rand", 10, 15)
    print(g_1)
    print(g_2)

if __name__ == '__main__':
  main()
