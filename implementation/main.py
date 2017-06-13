import tensorflow as tf

from datasets import DataSets
from pagerank.NumericPageRank import NumericPageRank
from graph.Graph import Graph
from utils import Utils


def main():
    reset_probability = 0.85

    graph_edges = DataSets.followers()

    with tf.Session() as sess:
        graph = Graph(sess, graph_edges, "G1")

        pr = NumericPageRank(sess, "PR_1", graph, reset_probability)

        print(pr.page_rank_vector(convergence=0.001))

        Utils.save_ranks('logs/test.csv', pr.ranks())


if __name__ == '__main__':
    main()
