import tensorflow as tf
import numpy as np
from datasets import DataSets
from graph.graph_sparsifier import GraphSparsifier
from pagerank.numeric_algebraic_page_rank import NumericAlgebraicPageRank
from pagerank.numeric_iterative_page_rank import NumericIterativePageRank
from pagerank.numeric_page_rank import NumericPageRank
from graph.graph import Graph
from utils import Utils
import timeit


def main():
    beta = 0.85

    wiki_vote_edges_np = DataSets.wiki_vote()
    followers_edges_np = DataSets.followers()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/.')

        graph_stream = Graph(sess, "Gfollowers_stream",
                             n=int(followers_edges_np.max(axis=0).max() + 1),
                             writer=writer)
        pr_stream = NumericIterativePageRank(sess, "PRs", graph_stream, beta)

        for r in followers_edges_np:
            start_time = timeit.default_timer()
            graph_stream.append(r[0], r[1])
            elapsed = timeit.default_timer() - start_time
            print(elapsed)
            print(pr_stream.ranks(convergence=0.0001))

        print(graph_stream)

        '''
        pr_1 = NumericPageRank(sess, "PR_1", followers_g, beta)
        ranks_1 = pr_1.ranks()
        print(ranks_1)
        Utils.save_ranks('logs/log_exact.csv', ranks_1)

        pr_2 = NumericIterativePageRank(sess, "PR_2", followers_g, beta)
        ranks_2 = pr_2.ranks(convergence=0.0001)
        print(ranks_2)
        Utils.save_ranks('logs/log_iterative.csv', ranks_1)

        graph_sparsifier = GraphSparsifier(sess, "G2", edges_np, 0.8)
        pr_3 = NumericIterativePageRank(sess, "PR_3", graph_sparsifier,
                                        beta)
        ranks_3 = pr_3.ranks(convergence=0.001)
        Utils.save_ranks('logs/log_iterative_sparsifier.csv', ranks_3)
        '''
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
