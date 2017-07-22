import numpy as np
import tensorflow as tf

from tfgraph.utils.callbacks.update_edge_notifier import UpdateEdgeNotifier
from tfgraph.utils.tensorflow_object import TensorFlowObject, TF_type


def __str__(self) -> str:
  return str(self.run_tf(self.L_tf))


class Graph(TensorFlowObject, UpdateEdgeNotifier):
  """ Graph class implemented in the top of TensorFlow.

  The class codifies the graph using an square matrix of 2-D shape and
  provides functionality operating with this matrix.

  Attributes:
    sess (:obj:`tf.Session`): This attribute represents the session that runs
      the TensorFlow operations.
    name (str): This attribute represents the name of the object in TensorFlow's
     op Graph.
    writer (:obj:`tf.summary.FileWriter`): This attribute represents a
      TensorFlow's Writer, that is used to obtain stats. The default value is
      `None`.
    _listeners (:obj:`set`): The set of objects that will be notified when an
      edge modifies it weight.
    n (int): Represents the cardinality of the vertex set as Python `int`.
    n_tf (:obj:`tf.Tensor`): Represents the cardinality of the vertex set as 0-D
      Tensor.
    m (int): Represents the cardinality of the edge set as Python `int`.
    A_tf (:obj:`tf.Tensor`): Represents the Adjacency matrix of the graph as
      2-D Tensor with shape [n,n].
    out_degrees_tf (:obj:`tf.Tensor`): Represents the out-degrees of the
      vertices of the graph as 2-D Tensor with shape [n, 1]
    in_degrees_tf (:obj:`tf.Tensor`): Represents the in-degrees of the vertices
      of the graph as 2-D Tensor with shape [1, n]

  """

  def __init__(self, sess: tf.Session, name: str,
               writer: tf.summary.FileWriter = None,
               edges_np: np.ndarray = None, n: int = None,
               is_sparse: bool = False) -> None:
    """ Class Constructor of the Graph

    This method is called to construct a Graph object. This block of code
    initializes all the variables necessaries for this class to properly works.

    This class can be initialized using an edge list, that fill the graph at
    this moment, or can be construct it from the cardinality of vertices set
    given by `n` parameter.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      writer (:obj:`tf.summary.FileWriter`, optional): This attribute represents
        a TensorFlow's Writer, that is used to obtain stats. The default value
        is `None`.
      edges_np (:obj:`np.ndarray`, optional): The edge set of the graph codifies
        as `edges_np[:,0]` represents the sources and `edges_np[:,1]` the
        destinations of the edges. The default value is `None`.
      n (int, optional): Represents the cardinality of the vertex set. The
        default value is `None`.
      is_sparse (bool, optional): Use sparse Tensors if it's set to `True`. The
        default value is False` Not implemented yet. Show the Todo for more
        information.

    Todo:
      * Implement variables as sparse when it's possible. Waiting to
        TensorFlow for it.

    """
    TensorFlowObject.__init__(self, sess, name, writer, is_sparse)
    UpdateEdgeNotifier.__init__(self)

    if edges_np is not None:
      if n is not None:
        self.n = max(n, int(edges_np.max(axis=0).max() + 1))
      else:
        self.n = int(edges_np.max(axis=0).max() + 1)
      self.m = int(edges_np.shape[0])
      A_init = tf.scatter_nd(edges_np.tolist(), self.m * [1.0],
                             [self.n, self.n])
    elif n is not None:
      self.n = n
      self.m = 0
      A_init = tf.zeros([self.n, self.n])
    else:
      raise ValueError('Graph constructor must be have edges or n')

    self.n_tf = tf.Variable(float(self.n), tf.float32,
                            name=self.name + "_n")
    self.A_tf = tf.Variable(A_init, tf.float64,
                            name=self.name + "_A")
    self.out_degrees_tf = tf.Variable(
      tf.reduce_sum(self.A_tf, 1, keep_dims=True),
      name=self.name + "_d_out")
    self.in_degrees_tf = tf.Variable(
      tf.reduce_sum(self.A_tf, 0, keep_dims=True),
      name=self.name + "_d_in")
    self.run_tf(tf.variables_initializer([self.A_tf, self.n_tf]))
    self.run_tf(tf.variables_initializer([
      self.out_degrees_tf, self.in_degrees_tf]))

  def __str__(self) -> str:
    """ Transforms the graph to a string.

    This method is used to print the graph on the command line. It codifies the
    laplacian matrix of the graph as string.

    Returns:
      (str): representing the laplacian matrix to visualize it.

    """
    return str(self.run_tf(self.L_tf))

  @property
  def L_tf(self) -> tf.Tensor:
    """ This method returns the Laplacian of the graph.

        The method generates a 2-D Array containing the laplacian matrix of the
        graph

        Returns:
          (:obj:`tf.Tensor`): A 2-D Tensor with [n,n] shape where n is the
            cardinality of the vertex set

        """
    return tf.diag(self.out_degrees_tf_vector) - self.A_tf

  @property
  def is_not_sink_tf(self) -> tf.Tensor:
    """ This method returns if a vertex is a sink vertex as vector.

    The method generates a 1-D Tensor containing the boolean values that
    indicates if the vertex at position `i` is a sink vertex.

    Returns:
      (:obj:`tf.Tensor`): A 1-D Tensor with the same length as cardinality
        of the vertex set.

    """
    return tf.not_equal(self.out_degrees_tf_vector, 0)

  def is_not_sink_tf_vertex(self, vertex: int) -> TF_type:
    """ This method returns if a vertex is a sink vertex as vector.

    The method generates a 1-D Tensor containing the boolean values that
    indicates if the vertex at position `i` is a sink vertex.

    Args:
      vertex (int): The index of the vertex that wants to know if is sink.

    Returns:
      (:obj:`tf.Tensor`): A 0-D Tensor that represents if a vertex is a sink
        vertex

    """
    return tf.not_equal(
      tf.reshape([self.out_degrees_tf_vertex(vertex)], [1]), 0)

  @property
  def out_degrees_np(self) -> np.ndarray:
    """ This method returns the degree of all vertex as vector.

    The method generates a 1-D Array containing the out-degree of the vertex `i`
    at position `i`

    Returns:
      (:obj:`np.ndarray`): A 1-D Array with the same length as cardinality of the
        vertex set.

    """
    return self.run_tf(self.out_degrees_tf)

  def out_degrees_tf_vertex(self, vertex: int) -> tf.Tensor:
    """ This method returns the degree of all vertex as vector.

    The method generates a 0-D Array containing the out-degree of the vertex i.

    Args:
      vertex (int): The index of the vertex that wants the degree.

    Returns:
      (:obj:`np.ndarray`): A 1-D Array with the same length as cardinality of the
        vertex set.

    """
    return tf.gather(self.out_degrees_tf, [vertex])

  @property
  def edge_list_tf(self) -> tf.Tensor:
    """ Method that returns the edge set of the graph as list.

    This method return all the edges of the graph codified as 2-D matrix in
    which the first dimension represents each edge and second dimension the
    source and destination vertices of each edge.

    Returns:
      (:obj:`tf.Tensor`): A 2-D Tensor with the he same length as cardinality of
       the edge set in the first dimension and 2 in the second.

    """
    return tf.cast(tf.where(tf.not_equal(self.A_tf, 0)), tf.int64)

  @property
  def edge_list_np(self) -> np.ndarray:
    """ Method that returns the edge set of the graph as list.

    This method return all the edges of the graph codified as 2-D matrix in
    which the first dimension represents each edge and second dimension the
    source and destination vertices of each edge.

    Returns:
      (:obj:`np.ndarray`): A 2-D Array with the he same length as cardinality of
        the edge set in the first dimension and 2 in the second.

    """
    return self.run_tf(self.edge_list_tf)

  @property
  def L_pseudo_inverse_tf(self) -> tf.Tensor:
    """ Method that returns the pseudo inverse of the Laplacian matrix.

    This method calculates the pseudo inverse matrix of the Laplacian of the
    Graph. It generates a matrix of the same shape as the Laplacian matrix, i.e.
    [n, n] where n is the cardinality of the vertex set.

    Returns:
      (:obj:`tf.Tensor`): A 2-D square Tensor with the he same length as
        cardinality of the vertex set representing the laplacian pseudo inverse.

    """
    return tf.py_func(np.linalg.pinv, [self.L_tf], tf.float32)

  def A_tf_vertex(self, vertex: int) -> tf.Tensor:
    """ Method that returns the adjacency of an individual vertex.

    This method extracts the corresponding row referred to the `vertex` passed
    as parameter. It constructs a vector that contains the weight of the edge
    between `vertex` (obtained as parameter) and the vertex at position `i` in
    the vector.

    Args:
      vertex (int): The index of the vertex that wants the degree.

    Returns:
      (:obj:`tf.Tensor`): A 1-D Tensor with the same length as the cardinality
        of the vertex set.

    """
    return tf.gather(self.A_tf, [vertex])

  @property
  def in_degrees_np(self) -> np.ndarray:
    """ This method returns the in-degree of all vertex as vector.

    The method generates a 1-D Array containing the in-degree of the vertex `i`
    at position `i`

    Returns:
      (:obj:`np.ndarray`): A 1-D Array with the same length as cardinality of the
        vertex set.

    """
    return self.run_tf(self.in_degrees_tf)

  @property
  def in_degrees_tf_vector(self):
    """ The in-degrees of the vertices of the graph

    Method that returns the in-degrees of the vertices of the graph as 1-D
    Tensor with shape [n]

    Returns:
      (:obj:`tf.Tensor`): A 1-D Tensor with the same length as the cardinality
        of the vertex set.

    """
    return tf.reshape(self.in_degrees_tf, [self.n])

  @property
  def out_degrees_tf_vector(self):
    """ The out-degrees of the vertices of the graph

    Method that returns the out-degrees of the vertices of the graph as 1-D
    Tensor with shape [n]

    Returns:
      (:obj:`tf.Tensor`): A 1-D Tensor with the same length as the cardinality
      of the vertex set.

    """
    return tf.reshape(self.out_degrees_tf, [self.n])

  def append(self, src: int, dst: int) -> None:
    """ Append an edge to the graph.

    This method process an input edge adding it to the graph updating all the
    variables necessaries to maintain the graph in correct state.

    Args:
      src (int): The id of the source vertex of the edge.
      dst (int): The id of the destination vertex of the edge.

    Returns:
      This method returns nothing.

    """
    if src and dst is None:
      raise ValueError(
        "tfgraph and dst must not be None ")
    self.run_tf([tf.scatter_nd_add(self.A_tf, [[src, dst]], [1.0]),
                 tf.scatter_nd_add(self.out_degrees_tf, [[src, 0]], [1.0]),
                 tf.scatter_nd_add(self.in_degrees_tf, [[0, dst]], [1.0])])
    self.m += 1
    self._notify(np.array([src, dst]), 1)

  def remove(self, src: int, dst: int) -> None:
    """ Remove an edge to the graph.

    This method process an input edge deleting it to the graph updating all the
    variables necessaries to maintain the graph in correct state.

    Args:
      src (int): The id of the source vertex of the edge.
      dst (int): The id of the destination vertex of the edge.

    Returns:
      This method returns nothing.

    """
    if src and dst is None:
      raise ValueError(
        "tfgraph and dst must not be None ")
    self.run_tf([tf.scatter_nd_add(self.A_tf, [[src, dst]], [-1.0]),
                 tf.scatter_nd_add(self.out_degrees_tf, [[src, 0]], [-1.0]),
                 tf.scatter_nd_add(self.in_degrees_tf, [[0, dst]], [-1.0])])
    self.m -= 1
    self._notify(np.array([src, dst]), -1)
