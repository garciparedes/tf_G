import tensorflow as tf

from datasets import DataSets
from pagerank.NumericPageRank import NumericPageRank

from utils import Utils


def main():
    reset_probability = 0.85

    graph_edges = DataSets.followers()

    with tf.Session() as sess:
        pr = NumericPageRank(sess, graph_edges, reset_probability)

        print(pr.page_rank_vector(convergence=0.001))
        #print(sess.run(tf.reduce_sum(pr.page_rank_vector(convergence=0.01))))

        Utils.save('logs/test.csv', pr.ranks()+1)


if __name__ == '__main__':
    main()
