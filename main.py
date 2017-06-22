import tensorflow as tf
import numpy as np
from src.graph.graph_constructor import GraphConstructor
from src.pagerank.numeric_algebraic_pagerank import NumericAlgebraicPageRank
from src.pagerank.numeric_iterative_pagerank import NumericIterativePageRank
from src.utils.datasets import DataSets
import timeit


def main():
    beta = 0.85
    convergence = 0.001

    wiki_vote_edges_np = DataSets.wiki_vote()
    followers_edges_np = DataSets.followers()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/.')

        g_followers = GraphConstructor.from_edges(sess, "Gfollowers",
                                                  followers_edges_np, writer)
        pr_followers = NumericIterativePageRank(sess, "PR1",g_followers, beta)
        
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
        
        print(g_followers)
        print(pr_followers.ranks(convergence=convergence))
        print(g_followers.m)

        g_sparse = GraphConstructor.as_other_sparsifier(sess, g_followers, 0.9)
        pr_sparse = NumericIterativePageRank(sess, "PR2",
                                             g_sparse, beta)
        print(g_sparse.m)
        print(pr_sparse.ranks(convergence=convergence))

        '''
        print(GraphConstructor.unweighted_random(sess, "GRandom", 10 ** 2,
                                                 10 ** 3, writer=writer)
        '''
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
