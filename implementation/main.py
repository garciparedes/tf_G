import tensorflow as tf

from datasets import DataSets
from pagerank.NumericPageRank import NumericPageRank

from utils import Utils


def main():
    steps = 7
    reset_probability = 0.85

    graph_edges = DataSets.followers()
    graph_edges -= 1

    with tf.Session() as sess:
        pr = NumericPageRank(sess, graph_edges, reset_probability)

        print(pr.page_rank_vector(steps))

        Utils.save('logs/test.csv', pr.ranks())


if __name__ == '__main__':
    main()
