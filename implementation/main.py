import tensorflow as tf

from datasets import DataSets
from graph.graph_sparsifier import GraphSparsifier
from pagerank.numeric_iterative_page_rank import NumericIterativePageRank
from pagerank.numeric_page_rank import NumericPageRank
from graph.graph import Graph
from utils import Utils


def main():
    reset_probability = 0.85

    graph_edges = DataSets.wiki_vote()

    with tf.Session() as sess:
        graph = Graph(sess, graph_edges, "G1")

        pr_1 = NumericIterativePageRank(sess, "PR_1", graph, reset_probability)
        ranks_1 = pr_1.ranks(convergence=0.001)
        Utils.save_ranks('logs/wiki_vote_log.csv', ranks_1)

        '''
        graph_sparsifier = GraphSparsifier(sess, graph_edges, 0.8, "G2")
        pr_2 = NumericIterativePageRank(sess, "PR_2", graph_sparsifier,
                                        reset_probability)
        ranks_2 = pr_2.ranks(convergence=0.001)
        Utils.save_ranks('logs/wiki_vote_log_sparsifier.csv', ranks_2)

        tf.summary.FileWriter('logs/.', sess.graph)
        '''

if __name__ == '__main__':
    main()
