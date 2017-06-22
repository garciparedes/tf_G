import tensorflow as tf
import numpy as np
import timeit
from pprint import pprint

from src.graph.graph_constructor import GraphConstructor
from src.pagerank.numeric_algebraic_pagerank import NumericAlgebraicPageRank
from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.pagerank.numeric_pagerank import NumericPageRank
from src.utils.datasets import DataSets


def main():
    beta = 0.85
    convergence = 0.001

    wiki_vote_edges_np = DataSets.wiki_vote()
    followers_edges_np = DataSets.followers()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/.')

        g_followers = GraphConstructor.from_edges(sess, "Gfollowers",
                                                  followers_edges_np, writer)
        pr_followers_iter = NumericIterativePageRank(sess, "PR_iter",
                                                     g_followers,
                                                     beta)
        pr_followers_alge = NumericAlgebraicPageRank(sess, "PR_alge",
                                                     g_followers,
                                                     beta)

        '''
        g_followers_updateable = GraphConstructor.empty(sess, "Gfollowers",
                                                  7, writer)
        pr_followers_updateable = NumericIterativePageRank(sess, "PRfollowers",
                                                           g_followers_updateable, beta)
        for r in followers_edges_np:
            start_time = timeit.default_timer()
            g_followers_updateable.append(r[0],r[1])
            print(timeit.default_timer() - start_time)
            print()
            writer.add_graph(sess.graph)
        g_followers_updateable.remove(followers_edges_np[0,0], followers_edges_np[0,1])
        g_followers_updateable.append(followers_edges_np[0,0], followers_edges_np[0,1])
        '''
        pprint(pr_followers_alge.ranks().tolist())
        # pprint(pr_followers_alge.ranks_by_id().tolist())
        # print(pr_followers_iter.ranks(convergence=convergence))
        # print(pr_followers_alge.error_rank_compare_np(pr_followers_iter))


        g_sparse = GraphConstructor.as_other_sparsifier(sess, g_followers, 0.95)

        pr_sparse = NumericAlgebraicPageRank(sess, "PR_sparse",
                                             g_sparse, beta)

        pprint(pr_sparse.ranks().tolist())

        # pprint(pr_sparse.ranks_by_rank(convergence=convergence).tolist())
        print(pr_followers_alge.error_ranks_compare_np(pr_sparse, k=100))
        print(g_followers.m)
        print(g_sparse.m)

        '''
        print(GraphConstructor.unweighted_random(sess, "GRandom", 10 ** 2,
                                                 10 ** 3, writer=writer)
        '''
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
