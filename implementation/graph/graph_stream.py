from datasets import DataSets
from graph.graph import Graph


class GraphStream(Graph):
    def __init__(self, sess, name):
        Graph.__init__(self, sess, name, DataSets.empty())

    def append(self, src, dst):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.edges_df = self.edges_df.append({'src': src, 'dst': dst},
                                             ignore_index=True)

    def remove(self, src=None, dst=None):
        if src and dst is None:
            raise ValueError("src and dst must not be None ")
        self.edges_df = self.edges_df[(self.edges_df['src'] != src) &
                                      (self.edges_df['dst'] != dst)]
