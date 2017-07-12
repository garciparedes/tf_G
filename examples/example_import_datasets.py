#!/usr/bin/python3

import tensorflow as tf
import numpy as np

import tf_G


def path() -> str:
  return "./../datasets"


def name_to_default_path(name: str) -> str:
  return path() + '/' + name + '/' + name + ".csv"


def compose_from_name(name: str, index_decrement: bool) -> np.ndarray:
  return tf_G.DataSets.compose_from_path(name_to_default_path(name),
                                         index_decrement)


def followers(index_decrement: bool = True) -> np.ndarray:
  return compose_from_name('followers', index_decrement)


def wiki_vote(index_decrement: bool = True) -> np.ndarray:
  return compose_from_name('wiki-Vote', index_decrement)


def p2p_gnutella08(index_decrement: bool = False) -> np.ndarray:
  return compose_from_name('p2p-gnutella08', index_decrement)


def main():
  with tf.Session() as sess:
    print(tf_G.GraphConstructor.from_edges(sess, "G_followers", followers()))

    print(tf_G.GraphConstructor.from_edges(sess, "G_p2p", p2p_gnutella08()))

    print(tf_G.GraphConstructor.from_edges(sess, "G_wiki", wiki_vote()))


if __name__ == '__main__':
  main()
