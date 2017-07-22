import numpy as np

from tfgraph.utils.callbacks.update_edge_listener import UpdateEdgeListener


class UpdateEdgeNotifier:
  """ This class is used to notify another classes that a change in graph edges.

  The graph (or another class that wants to notify an edge change) inherits
  this class and when an edge changes it will notify this change to all the
  attached objects.

  The objects attached to this class need to implement
  `update_edge(edge,change)` method.

  Attributes:

    _listeners (:obj:`set`): The set of objects that will be notified when an
      edge modifies it weight.

  """

  def __init__(self):
    """ Constructor of UpdateEdgeNotifier

    The set of listeners is initialised.

    """
    self._listeners = set()

  def attach(self, listener: UpdateEdgeListener):
    """ Method to attach objects from this class notifications.

    Args:
      listener (:obj:`tfgraph.UpdateEdgeListener`): An object that will start being
        notified when the graph changes its edge set.

    Returns:
      This method returns nothing.

    """
    self._listeners.add(listener)

  def detach(self, listener: UpdateEdgeListener):
    """ Method to detach objects from this clas notifications.

    Args:
      listener (:obj:`tfgraph.UpdateEdgeListener`): An object that will stop being
        notified when the graph changes its edge set.

    Returns:
      This method returns nothing.

    """
    self._listeners.discard(listener)

  def _notify(self, edge: np.ndarray, change: float):
    """ The private method that is used internally to notify the changes to
        attached classes.

    This method will broadcast the `change` of the `edge` to all the objects
    attached to this class.

    Args:
      edge (:obj:`np.ndarray`): The vector of shape [2] that represent and edge
        being edge[0] the source vertex and edge[1] the destination vertex.
      change (float): The variation in the edge weight.

    Returns:
      This method returns nothing.

    """
    for listener in self._listeners:
      listener.update_edge(edge, change)
