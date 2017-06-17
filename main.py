import tensorflow as tf

from src.datasets import DataSets
from src.graph.graph import Graph
from src.graph.random_graph_generator import RandomGraph
from src.pagerank.numeric_iterative_page_rank import NumericIterativePageRank


def main():
    beta = 0.85
    convergence = 0.001

    wiki_vote_edges_np = DataSets.wiki_vote()
    followers_edges_np = DataSets.followers()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/.')

        g_followers = Graph(sess, "Gfollowers", edges_np=followers_edges_np,
                            writer=writer)
        pr_followers = NumericIterativePageRank(sess, "PRfollowers",
                                                g_followers, beta)
        print(g_followers)
        print(pr_followers.ranks(convergence=convergence, personalized=[2]))

        # gs_followers = g_followers.sparsifier(alpha=0.9)

        '''
        print(RandomGraph.generate(sess, "GRandom", 10, 10, writer=writer))
        
        g_wiki_vote = Graph(sess, "Gwikivote", edges_np=wiki_vote_edges_np,
                            writer=writer)
        pr_wiki_vote = NumericIterativePageRank(sess, "PRwikivote",
                                                g_wiki_vote, beta)
        print(g_wiki_vote)
        print(pr_wiki_vote.ranks(convergence=convergence))
        '''

        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
