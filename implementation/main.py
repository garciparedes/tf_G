import tensorflow as tf

from datasets import DataSets
from graph.graph_sparsifier import GraphSparsifier
from pagerank.numeric_page_rank import NumericPageRank
from graph.graph import Graph
from utils import Utils


def main():
    reset_probability = 0.85

    graph_edges = DataSets.wiki_vote()

    with tf.Session() as sess:
        graph = Graph(sess, graph_edges, "G1")

        pr = NumericPageRank(sess, "PR_1", graph, reset_probability)
        ranks = pr.ranks(convergence=0.001)
        print(ranks)
        Utils.save_ranks('logs/log.csv', ranks)

        graph_sparsifier = GraphSparsifier(sess, graph_edges, 0.9, "H1")
        pr = NumericPageRank(sess, "PR_1", graph_sparsifier, reset_probability)
        ranks = pr.ranks(convergence=0.001)
        print(ranks)
        Utils.save_ranks('logs/log_sparsifier.csv', ranks)

        tf.summary.FileWriter('logs/.', sess.graph)


if __name__ == '__main__':
    main()
