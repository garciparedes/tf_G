import warnings
from typing import List

import numpy as np
import tensorflow as tf
from tfgraph.utils.math.vector_norm import VectorNorm

from tfgraph.algorithms.pagerank import Transition
from tfgraph.graph import Graph
from tfgraph.utils.callbacks.update_edge_listener import UpdateEdgeListener
from tfgraph.utils.math.convergence_criterion import ConvergenceCriterion
from tfgraph.utils.tensorflow_object import TensorFlowObject
from tfgraph.utils.utils import Utils


class PageRank(TensorFlowObject, UpdateEdgeListener):
  """ PageRank base class.

  This class model the PageRank algorithm as Abstract Class containing all
  methods that the heir classes need to implements. Also, this class provides
  a set of attributes that helps to implement the algorithm.

  The PageRank algorithm calculates the rank of each vertex in a graph based on
  the relational structure from them and giving more importance to the vertices
  that connects with edges to vertices with very high in-degree recursively.

  This class depends on the TensorFlow library, so it's necessary to install it
  to properly work.

  Attributes:
    sess (:obj:`tf.Session`): This attribute represents the session that runs
      the TensorFlow operations.
    name (str): This attribute represents the name of the object in TensorFlow's
      op Graph.
    G (:obj:`tfgraph.Graph`): The graph on witch it will be calculated the
      algorithm. It will be treated as Directed Weighted Graph.
    beta (float): The reset probability of the random walks, i.e. the
      probability that a user that surfs the graph an decides to jump to another
      vertex not connected to the current.
    T (:obj:`tfgraph.Transition`): The transition matrix that provides the
      probability distribution relative to the walk to another node of the graph.
    v (:obj:`tf.Variable`): The stationary distribution vector. It contains the
      normalized probability to stay in each vertex of the graph. So represents
      the PageRank ranking of the graph.
    writer (:obj:`tf.summary.FileWriter`): This attribute represents a
      TensorFlow's Writer, that is used to obtain stats.
    is_sparse (bool): Use sparse Tensors if it's set to True. Not
      implemented yet.

  """

  def __init__(self, sess: tf.Session, name: str, graph: Graph, beta: float,
               T: Transition, writer: tf.summary.FileWriter = None,
               is_sparse: bool = False) -> None:
    """ The constructor of the class.

    This method initializes all the attributes needed to compute the PageRank
    of the graph.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      beta (float): The reset probability of the random walks, i.e. the
        probability that a user that surfs the graph an decides to jump to
        another vertex not connected to the current.
      T (:obj:`tfgraph.Transition`): The transition matrix that provides the
        probability distribution relative to the walk to another node of the
        graph.
      v (:obj:`tf.Variable`): The stationary distribution vector. It contains
        the normalized probability to stay in each vertex of the graph. So
        represents the PageRank ranking of the graph.
      writer (:obj:`tf.summary.FileWriter`): This attribute represents a
        TensorFlow's Writer, that is used to obtain stats.
      is_sparse (bool): Use sparse Tensors if it's set to True. Not
        implemented yet.

    """
    TensorFlowObject.__init__(self, sess, name, writer=writer,
                              is_sparse=is_sparse)
    UpdateEdgeListener.__init__(self)

    self.beta = beta
    self.T = T
    self.T.attach(self)
    self.v = tf.Variable(tf.fill([1, self.T.G.n], tf.pow(self.T.G.n_tf, -1)),
                         name=self.T.G.name + "_" + self.name + "_v")
    self.run_tf(tf.variables_initializer([self.v]))

  def error_vector_compare_tf(self, other_pr: 'PageRank',
                              k: int = -1) -> tf.Tensor:
    """ The comparison method between two PageRank algorithm results.

    This method compares the `self` PageRank with another one passed as
    parameter of the function. The comparison is based on the difference of the
    Norm One of each `v` vector.

    The method also provides a `k` parameter as option to base the comparison
    only the `k` better ranked vertices.

    Args:
      other_pr (:obj:`tfgraph.PageRank`): Another PageRank object to compare the
        resulting ranking.
      k (int, optional): An additional parameter that allows to base the
        comparison only on the `k` better vertices. Not implemented yet.

    Returns:
      (:obj:`tf.Tensor`): A `tf.Tensor` with 0-D shape, that represents the
        difference between the two rankings using the Norm One.

    Todo:
      * Implement ranking based only on the `k` better ranked vertices.

    """
    if 0 < k < self.T.G.n - 1:
      if 0 < k < self.T.G.n - 1:
        warnings.warn('k-best error comparison not implemented yet')

    return tf.reshape(
      VectorNorm.ONE(tf.subtract(self.v, other_pr.v)), [])

  def error_vector_compare_np(self, other_pr: 'PageRank',
                              k: int = -1) -> np.ndarray:
    """ The comparison method between two PageRank algorithm results.

    This method compares the `self` PageRank with another one passed as
    parameter of the function. The comparison is based on the difference of the
    Norm One of each `v` vector.

    The method also provides a `k` parameter as option to base the comparison
    only the `k` better ranked vertices.

    Args:
      other_pr (:obj:`tfgraph.PageRank`): Another PageRank object to compare the
        resulting ranking.
      k (int, optional): An additional parameter that allows to base the
        comparison only on the `k` better vertices. Not implemented yet.

    Returns:
      (:obj:`np.ndarray`): A `np.ndarray` with 0-D shape, that represents the
        difference between the two rankings using the Norm One.

    """
    return self.run_tf(self.error_vector_compare_tf(other_pr, k))

  def pagerank_vector_tf(self, convergence: float = 1.0, steps: int = 0,
                         topics: List[int] = None,
                         topics_decrement: bool = False,
                         c_criterion=ConvergenceCriterion.ONE) -> tf.Tensor:
    """ The Method that runs the PageRank algorithm

    This method generates a TensorFlow graph of operations needed to calculate
    the PageRank Algorithm and sets to it different parameters passed as
    parameters.

    This method acts as interface between the algorithm and the external
    classes, so it contains a set of parameters that in some implementations of
    PageRank algorithms will not be needed. All the parameters is defined as
    optional for this reason.

    Args:
      convergence (float, optional): A float between 0 and 1 that represents
        the convergence rate that allowed to finish the iterative
        implementations of the algorithm to accept the solution. It has more
        preference than the `steps` parameter. Default to `1.0`.
      steps (int, optional): A positive integer that sets the number of
        iterations that the iterative implementations will run the algorithm
        until finish. It has less preference than the `convergence` parameter.
        Default to `0`.
      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`.
      topics_decrement (bool, optional): If topics is not None and
        topics_decrement is `True` the topics will be casted to 0-Index. Default `
        to False`.
      c_criterion (:obj:`function`, optional): The function used to calculate if
        the Convergence Criterion of the iterative implementations is reached.
        Default to `tfgraph.ConvergenceCriterion.ONE`.

    Returns:
      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    if topics_decrement is True and topics is not None:
      topics = [item - 1 for item in topics]

    if 0.0 < convergence < 1.0:
      return self._pr_convergence_tf(convergence, topics=topics,
                                     c_criterion=c_criterion)
    elif steps > 0:
      return self._pr_steps_tf(steps, topics=topics)
    else:
      return self._pr_exact_tf(topics=topics)

  def pagerank_vector_np(self, convergence: float = 1.0, steps: int = 0,
                         topics: List[int] = None,
                         c_criterion=ConvergenceCriterion.ONE) -> np.ndarray:
    """ The Method that runs the PageRank algorithm

    This method returns a Numpy Array that contains the result of running the
    PageRank algorithm customized by the parameters passed to it.

    This method acts as interface between the algorithm and the external
    classes, so it contains a set of parameters that in some implementations of
    PageRank algorithms will not be needed. All the parameters is defined as
    optional for this reason.

    Args:
      convergence (float, optional): A float between 0 and 1 that represents
        the convergence rate that allowed to finish the iterative
        implementations of the algorithm to accept the solution. It has more
        preference than the `steps` parameter. Default to `1.0`.
      steps (int, optional): A positive integer that sets the number of
        iterations that the iterative implementations will run the algorithm
        until finish. It has less preference than the `convergence` parameter.
        Default to `0`.
      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`.
      c_criterion (:obj:`function`, optional): The function used to calculate if
        the Convergence Criterion of the iterative implementations is reached.
        Default to `tfgraph.ConvergenceCriterion.ONE`.

    Returns:
      (:obj:`np.ndarray`): A 1-D `np.ndarray` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    return self.run_tf(
      self.pagerank_vector_tf(convergence, steps, topics,
                              c_criterion))

  def ranks_np(self, convergence: float = 1.0, steps: int = 0,
               topics: List[int] = None,
               topics_decrement: bool = False) -> np.ndarray:
    """ Generates a ranked version of PageRank results.

    This method returns the PageRank ranking of the graph sorted by the position
    of each vertex in the rank. So it generates a 2-D matrix with shape [n,2]
    where n is the cardinality of the vertex set of the graph, and at the first
    column it contains the index of vertex and the second column contains it
    normalized rank. The `i` row is referred to the vertex with `i` position in
    the rank.

    Args:
      convergence (float, optional): A float between 0 and 1 that represents
        the convergence rate that allowed to finish the iterative
        implementations of the algorithm to accept the solution. It has more
        preference than the `steps` parameter. Default to `1.0`.
      steps (int, optional): A positive integer that sets the number of
        iterations that the iterative implementations will run the algorithm
        until finish. It has less preference than the `convergence` parameter.
        Default to `0`.
      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`.
      topics_decrement (bool, optional): If topics is not None and
        topics_decrement is `True` the topics will be casted to 0-Index. Default `
        to False`.

    Returns:
      (:obj:`np.ndarray`): A 2-D `np.ndarray` than represents a sorted PageRank
        ranking of the graph.

    """
    self.pagerank_vector_tf(convergence, steps, topics, topics_decrement)
    ranks = tf.map_fn(
      lambda x: [x, tf.gather(tf.reshape(self.v, [self.T.G.n]), x)],
      tf.transpose(
        tf.py_func(Utils.ranked, [tf.scalar_mul(-1, self.v)], tf.int64)),
      dtype=[tf.int64, tf.float32])
    return np.concatenate(self.run_tf(ranks), axis=1)

  def _pr_convergence_tf(self, convergence: float, topics: List[int] = None,
                         c_criterion=ConvergenceCriterion.ONE) -> tf.Tensor:
    """ Abstract method to implement a iterative version of PageRank until
      convergence rate.


    This method runs the PageRank algorithm in iterative fashion a undetermined
    number of times bounded by the `convergence` rate and the 'c_criterion'
    criterion.

    Args:
      convergence (float): A float between 0 and 1 that represents
        the convergence rate that allowed to finish the iterative
        implementations of the algorithm to accept the solution. Default to
        `1.0`.
      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`.
      c_criterion (:obj:`function`, optional): The function used to calculate if
        the Convergence Criterion of the iterative implementations is reached.
        Default to `tfgraph.ConvergenceCriterion.ONE`.

    Returns:
      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    raise NotImplementedError(
      'subclasses must override page_rank_until_convergence()!')

  def _pr_steps_tf(self, steps: int, topics: List[int] = None) -> tf.Tensor:
    """ Abstract method to implement a iterative version of PageRank with fixed
      steps.

    This method runs the PageRank algorithm in iterative fashion a fixed number
    of times bounded by the `steps` parameter.

    Args:
      steps (int): A positive integer that sets the number of
        iterations that the iterative implementations will run the algorithm
        until finish. Default to `0`.
      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`.

    Returns:
      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    raise NotImplementedError(
      'subclasses must override page_rank_until_steps()!')

  def _pr_exact_tf(self, topics: List[int] = None) -> tf.Tensor:
    """ Abstract method to implement a exact version of PageRank.

    This method calculates the PageRank of the graph in exact mode.

    Args:
      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`.

    Returns:
      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    raise NotImplementedError(
      'subclasses must override page_rank_exact()!')

  def update_edge(self, edge: np.ndarray, change: float) -> None:
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
