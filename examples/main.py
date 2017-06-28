import tensorflow as tf
import numpy as np
import timeit

from tfg_big_data_algorithms.graph.graph import  Graph
from tfg_big_data_algorithms.graph.graph_constructor import GraphConstructor
from tfg_big_data_algorithms.pagerank.numeric_iterative_pagerank import \
    NumericIterativePageRank
from tfg_big_data_algorithms.pagerank.pagerank import PageRank
from tfg_big_data_algorithms.utils.datasets import DataSets


def main():
    beta: float = 0.85
    convergence: float = 0.01

    wiki_vote_edges_np: np.ndarray = DataSets.wiki_vote()
    followers_edges_np: np.ndarray = DataSets.followers()

    with tf.Session() as sess:
        writer: tf.summary.FileWriter = tf.summary.FileWriter(
            'examples/logs/tensorflow/.')

        g_followers: Graph = GraphConstructor.from_edges(sess, "Gfollowers",
                                                         followers_edges_np,
                                                         writer,
                                                         is_sparse=False)
        '''
        pr_followers_alge: PageRank = NumericAlgebraicPageRank(sess, "PR1",
                                                     g_followers,
                                                     beta)
        '''
        pr_followers_iter: PageRank = NumericIterativePageRank(sess, "PR1",
                                                               g_followers,
                                                               beta)
        '''
        pr_followers_random: PageRank = NumericRandomWalkPageRank(sess, "PR3",
                                                        g_followers,
                                                        beta)
        
        g_followers_updateable: Graph = GraphConstructor.empty(sess, "Gfollowers",
                                                  7, writer)
        pr_followers_updateable: PageRank = NumericIterativePageRank(sess, "PRfollowers",
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
        # a = pr_followers_alge.ranks_np()
        start_time: float = timeit.default_timer()
        b: np.ndarray = (pr_followers_iter.ranks_np(convergence=convergence))
        elapsed: float = timeit.default_timer() - start_time
        print(elapsed)
        '''
        start_time = timeit.default_timer()
        c = (pr_followers_random.ranks_np(convergence=convergence))
        elapsed = timeit.default_timer() - start_time
        print(elapsed)
        '''
        # print(a)
        print(b)
        # print(c)
        # print((pr_followers_alge.error_vector_compare_np(pr_followers_iter)))
        # print((pr_followers_alge.error_vector_compare_np(pr_followers_random)))
        # print(pr_followers_iter.error_vector_compare_np(pr_followers_random))
        # print(pr_followers_iter.ranks_np(convergence=convergence))
        # print(pr_followers_alge.error_ranks_compare_np(pr_followers_iter))


        # Utils.save_ranks("logs/csv/alge.csv",a)
        # Utils.save_ranks("logs/csv/iter.csv",b)
        # Utils.save_ranks("logs/csv/random.csv",c)
        '''
        g_sparse = GraphConstructor.as_other_sparsifier(sess, g_followers, 0.95)

        pr_sparse = NumericAlgebraicPageRank(sess, "PR_sparse",
                                             g_sparse, beta)

        pprint(pr_sparse.ranks_np().tolist())

        # pprint(pr_sparse.ranks_by_rank(convergence=convergence).tolist())
        print(pr_followers_alge.error_ranks_compare_np(pr_sparse, k=128))
        print(g_followers.m)
        print(g_sparse.m)

  
        print(GraphConstructor.unweighted_random(sess, "GRandom", 10 ** 2,
                                                 10 ** 3, writer=writer))
        '''
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
