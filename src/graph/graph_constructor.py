import numpy as np

from src.graph.graph import Graph


class GraphConstructor:
    @staticmethod
    def from_edges(sess, name, edges_np, writer=None):
        return Graph(sess, name, edges_np=edges_np, writer=writer)

    @staticmethod
    def empty(sess, name, n, writer=None):
        return Graph(sess, name, n=n, writer=writer)

    @staticmethod
    def random(sess, name, n, m, writer=None):
        edges_np = np.random.random_integers(0, n - 1, [m, 2])

        print(np.unique(edges_np))

        return Graph(sess, name, edges_np=edges_np, writer=writer)
