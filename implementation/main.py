import tensorflow as tf

from datasets import DataSets
from graph.graph_sparsifier import GraphSparsifier
from pagerank.numeric_iterative_page_rank import NumericIterativePageRank
from pagerank.numeric_page_rank import NumericPageRank
from graph.graph import Graph
from utils import Utils


def main():
    beta = 0.85

    graph_edges = DataSets.followers()

    with tf.Session() as sess:
        graph = Graph(sess, "G1", graph_edges)

        pr_1 = NumericPageRank(sess, "PR_1", graph, beta)
        ranks_1 = pr_1.ranks()
        print(ranks_1)
        Utils.save_ranks('logs/log_exact.csv', ranks_1)

        pr_2 = NumericIterativePageRank(sess, "PR_2", graph, beta)
        ranks_2 = pr_2.ranks(convergence=0.0001)
        print(ranks_2)
        Utils.save_ranks('logs/log_iterative.csv', ranks_1)

        '''
        graph_sparsifier = GraphSparsifier(sess, "G2", graph_edges, 0.8)
        pr_3 = NumericIterativePageRank(sess, "PR_3", graph_sparsifier,
                                        beta)
        ranks_3 = pr_3.ranks(convergence=0.001)
        Utils.save_ranks('logs/log_iterative_sparsifier.csv', ranks_3)

        '''
        tf.summary.FileWriter('logs/.', sess.graph)


if __name__ == '__main__':
    main()
