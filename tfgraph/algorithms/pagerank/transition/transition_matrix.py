import numpy as np
import tensorflow as tf

from tfgraph.algorithms.pagerank.transition.transition import Transition
from tfgraph.graph.graph import Graph


class TransitionMatrix(Transition):
  """ Transition Matrix Class

  This class implements the functionality of a 2-D matrix that represents the
  probability distribution of walk between the vertices of the graph.

  Attributes:
    sess (:obj:`tf.Session`): This attribute represents the session
      that runs the TensorFlow operations.
    name (str): This attribute represents the name of the object in TensorFlow's
      op Graph.
    writer (:obj:`tf.summary.FileWriter`): This attribute represents a
      TensorFlow's Writer, that is used to obtain stats.
    is_sparse (bool): Use sparse Tensors if it's set to True. Not implemented
      yet. Show the Todo.
    G (:obj:`tfgraph.Graph`):  The graph on which the transition is referred.
    transition (:obj:`tf.Variable`): The 2-D `tf.Tensor` with the same shape as
      adjacency matrix of the graph, that represents the probabilities to
      move from one vertex to another.

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
    Transition.__init__(self, sess=sess, name=name, graph=graph, writer=writer,
                        is_sparse=is_sparse)

    self.transition = tf.Variable(
      tf.where(self.G.is_not_sink_tf,
               tf.div(self.G.A_tf, self.G.out_degrees_tf),
               tf.fill([self.G.n, self.G.n], 1 / self.G.n)),
      name=self.name)
    self.run_tf(tf.variables_initializer([self.transition]))

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
    return self.transition

  def update_edge(self, edge: np.ndarray, change: float) -> None:
    """ The callback to receive notifications about edge changes in the graph.

     This method is called from the Graph when an addition or deletion is
     produced on the edge set. So probably is necessary to recompute the
     transition matrix.

     Args:
       edge (:obj:`np.ndarray`): A 1-D `np.ndarray` that represents the edge that
         changes in the graph, where `edge[0]` is the source vertex, and
         `edge[1]` the destination vertex.
       change (float): The variation of the edge weight. If the final value is
         0.0 then the edge is removed.

     Returns:
       This method returns nothing.

     """

    if change > 0.0:
      self.run_tf(tf.scatter_nd_update(
        self.transition, [[edge[0]]],
        tf.div(self.G.A_tf_vertex(edge[0]),
               self.G.out_degrees_tf_vertex(edge[0]))))
    else:
      self.run_tf(tf.scatter_nd_update(
        self.transition, [[edge[0]]],
        tf.where(self.G.is_not_sink_tf_vertex(edge[0]),
                 tf.div(self.G.A_tf_vertex(edge[0]),
                        self.G.out_degrees_tf_vertex(edge[0])),
                 tf.fill([1, self.G.n], tf.pow(self.G.n_tf, -1)))))
    self._notify(edge, change)
