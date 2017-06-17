from graph.graph import Graph
import numpy as np


class RandomGraph:
    @staticmethod
    def generate(sess, name, n, m, writer=None):
        edges_np = np.random.random_integers(0, n - 1, [m,2])

        print(np.unique(edges_np))

        return Graph(sess, name, edges_np=edges_np, writer=writer)
