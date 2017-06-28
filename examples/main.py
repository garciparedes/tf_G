import tensorflow as tf
import numpy as np
import timeit

import tfg_big_data_algorithms as tf_G


def main():
    beta: float = 0.85
    convergence: float = 0.01

    wiki_vote_np: np.ndarray = tf_G.DataSets.wiki_vote()
    followers_np: np.ndarray = tf_G.DataSets.followers()

    with tf.Session() as sess:
        writer: tf.summary.FileWriter = tf.summary.FileWriter(
            'logs/tensorflow/.')

        graph: tf_G.Graph = tf_G.GraphConstructor.from_edges(sess,
                                                             "Gfollowers",
                                                             followers_np,
                                                             writer,
                                                             is_sparse=False)
        '''
        pr_alge: tf_G.PageRank = tf_G.NumericAlgebraicPageRank(sess, "PR1",
                                                     graph,
                                                     beta)
        '''
        pr_iter: tf_G.PageRank = tf_G.NumericIterativePageRank(sess,
                                                               "PR1",
                                                               graph,
                                                               beta)
        '''
        pr_random: tf_G.PageRank = tf_G.NumericRandomWalkPageRank(sess, "PR3",
                                                        graph,
                                                        beta)
        '''
        g_updateable: tf_G.Graph = tf_G.GraphConstructor.empty(sess,
                                                               "Gfollowers",
                                                               7, writer)
        pr_updateable: tf_G.PageRank = tf_G.NumericIterativePageRank(sess,
                                                                     "PRfollowers",
                                                                     g_updateable,
                                                                     beta)

        '''
        for r in followers_np:
            start_time = timeit.default_timer()
            g_updateable.append(r[0],r[1])
            print(timeit.default_timer() - start_time)
            print()
            writer.add_graph(sess.graph)
        g_updateable.remove(followers_np[0,0], followers_np[0,1])
        g_updateable.append(followers_np[0,0], followers_np[0,1])
        '''
        # a = pr_alge.ranks_np()
        start_time: float = timeit.default_timer()
        b: np.ndarray = (pr_iter.ranks_np(convergence=convergence))
        elapsed: float = timeit.default_timer() - start_time
        print(elapsed)
        '''
        start_time = timeit.default_timer()
        c = (pr_random.ranks_np(convergence=convergence))
        elapsed = timeit.default_timer() - start_time
        print(elapsed)
        '''
        # print(a)
        print(b)
        # print(c)
        # print((pr_alge.error_vector_compare_np(pr_iter)))
        # print((pr_alge.error_vector_compare_np(pr_random)))
        # print(pr_iter.error_vector_compare_np(pr_random))
        # print(pr_iter.ranks_np(convergence=convergence))
        # print(pr_alge.error_ranks_compare_np(pr_iter))


        # Utils.save_ranks("logs/csv/alge.csv",a)
        # Utils.save_ranks("logs/csv/iter.csv",b)
        # Utils.save_ranks("logs/csv/random.csv",c)
        '''
        g_sparse = GraphConstructor.as_other_sparsifier(sess, graph, 0.95)

        pr_sparse = NumericAlgebraicPageRank(sess, "PR_sparse",
                                             g_sparse, beta)

        pprint(pr_sparse.ranks_np().tolist())

        # pprint(pr_sparse.ranks_by_rank(convergence=convergence).tolist())
        print(pr_alge.error_ranks_compare_np(pr_sparse, k=128))
        print(graph.m)
        print(g_sparse.m)

  
        print(GraphConstructor.unweighted_random(sess, "GRandom", 10 ** 2,
                                                 10 ** 3, writer=writer))
        '''
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
