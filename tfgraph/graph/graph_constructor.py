import tensorflow as tf
import numpy as np

from tfgraph.graph.graph import Graph
from tfgraph.graph.graph_sparsifier import GraphSparsifier


class GraphConstructor:
  """ Class that helps to construct a Graph object.

  This class contains a set of static methods that helps in the task of graph
  construction and initialization. It provides the generation of a Graph from
  an edge list, and allows to generate empty Graphs, sparsifier Graphs an also
  random Graphs.

  """

  @staticmethod
  def from_edges(sess: tf.Session, name: str, edges_np: np.ndarray,
                 writer: tf.summary.FileWriter = None,
                 is_sparse: bool = False) -> Graph:
    """ Generates a graph from a set of edges.

    This method acts as interface between the Graph constructor and the rest
    exterior.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      edges_np (:obj:`np.ndarray`): The edge set of the graph codifies as
        `edges_np[:,0]` represents the sources and `edges_np[:,1]` the
        destinations of the edges.
      writer (:obj:`tf.summary.FileWriter`, optional): This attribute represents
        a TensorFlow's Writer, that is used to obtain stats. The default value
        is `None`.
      is_sparse (bool, optional): Use sparse Tensors if it's set to True. Not
        implemented yet. Show the Todo. The default value is `False`.

    Returns:
      (:obj:`tfgraph.Graph`): A graph containing all the edges passed as input in
        `edges_np`.

    """
    return Graph(sess, name, edges_np=edges_np, writer=writer,
                 is_sparse=is_sparse)

  @staticmethod
  def empty(sess: tf.Session, name: str, n: int,
            writer: tf.summary.FileWriter = None,
            sparse: bool = False) -> Graph:

    """ Generates an empty Graph.

    This method generates an empty graph with the number of vertex fixed at the
    construction. The graph allows addition and deletion of edges.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      n (int): The cardinality of vertex set of the empty graph.
      writer (:obj:`tf.summary.FileWriter`, optional): This attribute represents
        a TensorFlow's Writer, that is used to obtain stats. The default value
        is `None`.
      is_sparse (bool, optional): Use sparse Tensors if it's set to True. Not
        implemented yet. Show the Todo. The default value is `False`.

    Returns:
      (:obj:`tfgraph.Graph`): A empty graph that allows additions and deletions of
        edges from vertex in the interval [0,n].

    """
    return Graph(sess, name, n=n, writer=writer, is_sparse=sparse)

  @staticmethod
  def empty_sparsifier(sess: tf.Session,
                       name: str,
                       n: int,
                       p: float,
                       writer: tf.summary.FileWriter = None,
                       is_sparse: bool = False) -> GraphSparsifier:
    """ Generates an empty Sparsifier Graph.

    This method generates an empty sparsifier graph with the number of vertex
    fixed at the construction. The graph allows addition and deletion of edges.
    The sparsifier means that the graph will not add all edges. Only a subset of
    it to improve the performance of the algorithms.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      n (int): The cardinality of vertex set of the empty graph.
      p (float): The picking probability value. It must be in the [0,1]
        interval.
      writer (:obj:`tf.summary.FileWriter`, optional): This attribute represents
        a TensorFlow's Writer, that is used to obtain stats. The default value
        is `None`.
      is_sparse (bool, optional): Use sparse Tensors if it's set to True. Not
        implemented yet. Show the Todo. The default value is `False`.

    Returns:
      (:obj:`tfgraph.GraphSparsifier`): A empty graph that allows additions and
        deletions of edges from vertex in the interval [0,n].

    """
    return GraphSparsifier(sess=sess, name=name, n=n, p=p, writer=writer,
                           is_sparse=is_sparse)

  @staticmethod
  def unweighted_random(sess: tf.Session, name: str, n: int, m: int,
                        writer: tf.summary.FileWriter = None,
                        is_sparse: bool = False) -> Graph:
    """ Generates a random unweighted graph.

    This method generates a random unweighted graph with `n` vertex and `m`
    edges. The edge set is generated using a uniform distribution.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      n (int): The cardinality of vertex set of the random graph.
      m (int): The cardinality of edge set of the random graph.
      writer (:obj:`tf.summary.FileWriter`, optional): This attribute represents
        a TensorFlow's Writer, that is used to obtain stats. The default value
        is `None`.
      is_sparse (bool, optional): Use sparse Tensors if it's set to True. Not
        implemented yet. Show the Todo. The default value is `False`.

    Returns:
      (:obj:`tfgraph.GraphSparsifier`): A empty graph that allows additions and
        deletions of edges from vertex in the interval [0,n].

    """
    if m > n * (n - 1):
      raise ValueError('m would be less than n * (n - 1)')
    edges_np = np.random.random_integers(0, n - 1, [m, 2])

    cond = True
    while cond:
      # remove uniques from: https://stackoverflow.com/a/16973510/3921457
      edges_np = np.concatenate((edges_np,
                                 np.random.random_integers(0, n - 1, [
                                   m - len(edges_np), 2])), axis=0)
      _, unique_idx = np.unique(np.ascontiguousarray(edges_np).view(
        np.dtype(
          (np.void, edges_np.dtype.itemsize * edges_np.shape[1]))),
        return_index=True)
      edges_np = edges_np[unique_idx]
      edges_np = edges_np[edges_np[:, 0] != edges_np[:, 1]]
      cond = len(edges_np) != m

    return Graph(sess, name, edges_np=edges_np, writer=writer,
                 is_sparse=is_sparse)

  @staticmethod
  def as_naive_sparsifier(sess: tf.Session, graph: Graph, p: float,
                          is_sparse: bool = False) -> Graph:
    """ Generates a sparsifier graph of the given graph.

    The method picks the edges with probability uniform probability `p` from
    edge set of the graph given as parameter. This does not provide any
    guarantee from the structure of the original graph.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      graph (:obj:`tfgraph.Graph`): The input graph to pick the edges
      p (float): The picking probability value. It must be in the [0,1]
        interval.
      is_sparse (bool): Use sparse Tensors if it's set to True. Not implemented
        yet.
    Returns:
      (:obj:`tfgraph.Graph`): The resulting graph with less edges than the original
        graph.

    """
    boolean_distribution = tf.less_equal(
      tf.random_uniform([graph.m], 0.0, 1.0), p)
    edges_np = graph.edge_list_np[sess.run(boolean_distribution)]
    return Graph(sess, graph.name + "_sparsifier",
                 edges_np=edges_np, is_sparse=is_sparse)

  @classmethod
  def as_sparsifier(cls, sess, graph: Graph, p: float, is_sparse=False):
    """ Generates a sparsifier graph from the given graph.

    The method picks the edges with probability uniform probability `p` from
    edge set of the graph given as parameter. The sparsifier uses an
    heuristic to picks the more edges from the vertices with big out_degree
    to try to maintain the structure of the graph.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      graph (:obj:`tfgraph.Graph`): The input graph to pick the edges.
      p (float): The picking probability value. It must be in the [0,1] interval.
      is_sparse (bool): Use sparse Tensors if it's set to True. Not implemented
        yet.

    Returns:
      (:obj:`tfgraph.Graph`): The resulting graph sparsifier with less edges than
        the original graph.

    """
    return GraphSparsifier(sess=sess, p=p, graph=graph, is_sparse=is_sparse)
