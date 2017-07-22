import tensorflow as tf
from tfgraph.utils.callbacks.update_edge_listener import UpdateEdgeListener
from tfgraph.graph.graph import Graph
from tfgraph.utils.callbacks.update_edge_notifier import UpdateEdgeNotifier
from tfgraph.utils.tensorflow_object import TensorFlowObject


class Transition(TensorFlowObject, UpdateEdgeNotifier, UpdateEdgeListener):
  """ Transition Base Class

  This class acts as base class of transition behavior between vertices of the
  graph. This class is used to use as base type that provides this functionality
  and also to store the common attributes that uses all Transition
  implementations.

  The heiress classes need to implement the `get_tf()` method that provides the
  transitions.

  Attributes:
    sess (:obj:`tf.Session`): This attribute represents the session that runs
      the TensorFlow operations.
    name (str): This attribute represents the name of the object in TensorFlow's
      op Graph.
    writer (:obj:`tf.summary.FileWriter`): This attribute represents a
      TensorFlow's Writer, that is used to obtain stats.
    is_sparse (bool): Use sparse Tensors if it's set to True. Not
      implemented yet. Show the Todo.
    _listeners (:obj:`set`): The set of objects that will be notified when an
      edge modifies it weight.
    G (:obj:`tfgraph.Graph`):  The graph on which the transition is referred.

  """

  def __init__(self, sess: tf.Session, name: str, graph: Graph,
               writer: tf.summary.FileWriter = None,
               is_sparse: bool = False) -> None:
    """ Constructor of the class.

    This method is called to create a new instance of Transition class.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      graph (:obj:`tfgraph.Graph`):  The graph on which the transition is referred.
      writer (:obj:`tf.summary.FileWriter`): This attribute represents a
        TensorFlow's Writer, that is used to obtain stats.
      is_sparse (bool): Use sparse Tensors if it's set to True. Not
        implemented yet. Show the Todo.

    """
    TensorFlowObject.__init__(self, sess, name + "_T", writer, is_sparse)
    UpdateEdgeNotifier.__init__(self)

    self.G = graph
    self.G.attach(self)

  def __call__(self, *args, **kwargs):
    """ The call method.

    In this case is used to retrieve the transition `tf.Tensor` that allows
    the algorithms to know the transition probabilities between each node. It
    calls the `get_tf()` method that is implemented by inner classes.

    Args:
      *args: The args of the `get_tf()` method.
      **kwargs: The kwargs of the `get_tf()` method.

    Returns:
      (:obj:`tf.Tensor`): A `tf.Tensor` that contains the distribution of
        transitions over vertices of the graph.

    """
    return self.get_tf(args, kwargs)

  def get_tf(self, *args, **kwargs):
    """ The method that returns the transition Tensor.

    This method will return the transition matrix of the graph.

    Args:
      *args: The args of the `get_tf()` method.
      **kwargs: The kwargs of the `get_tf()` method.

    Returns:
      (:obj:`tf.Tensor`): A `tf.Tensor` that contains the distribution of
        transitions over vertices of the graph.

    """
    raise NotImplementedError(
      'subclasses must override get_tf()!')
