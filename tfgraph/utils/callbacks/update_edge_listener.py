import numpy as np


class UpdateEdgeListener:
  """ Listen notifications related with a change in graph edges.

  The graph (or another class that wants to receive notifications from a edge
  change) inherits this class and when an edge changes it will receive
  notifications from the change in edge set of a graph.

  The classes that inherit this class need to implement
  `update_edge(edge,change)` method.

  """

  def update_edge(self, edge: np.ndarray, change: float):
    """ The callback to receive notifications about edge changes in the graph.

    This method is called from the Graph when an addition or deletion is
    produced on the edge set. So probably is necessary to recompute the PageRank
    ranking.


    Args:
      edge (:obj:`np.ndarray`): A 1-D `np.ndarray` that represents the edge that
        changes in the graph, where `edge[0]` is the source vertex, and
        `edge[1]` the destination vertex.
      change (float): The variation of the edge weight. If the final value is
        0.0 then the edge is removed.

    Returns:
      This method returns nothing.

    """
    raise NotImplementedError(
      'subclasses must override update_edge()!')
