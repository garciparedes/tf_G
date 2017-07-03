import warnings
from typing import List

import tensorflow as tf
import numpy as np

from tf_G.pagerank.transition.transition_reset_matrix import \
  TransitionResetMatrix
from tf_G.pagerank.pagerank import PageRank
from tf_G.utils.convergence_criterion import ConvergenceCriterion
from tf_G.graph.graph import Graph


class IterativePageRank(PageRank):
  """ The Iterative PageRank implementation.

  This class acts as the iterative algorithm to obtain the PageRank ranking of
  a graph.

  The PageRank algorithm calculates the rank of each vertex in a graph based on
  the relational structure from them and giving more importance to the vertices
  that connects with edges to vertices with very high in-degree recursively.

  This class depends on the TensorFlow library, so it's necessary to install it
  to properly work.

  Attributes:

    sess (:obj:`tf.Session`): This attribute represents the session
        that runs the TensorFlow operations.

    name (str): This attribute represents the name of the object in
      TensorFlow's op Graph.

    G (:obj:`tf_G.Graph`): The graph on witch it will be calculated the
      algorithm. It will be treated as Directed Weighted Graph.

    beta (float): The reset probability of the random walks, i.e. the
      probability that a user that surfs the graph an decides to jump to another
      vertex not connected to the current.

    T (:obj:`tf_G.Transition): The transition matrix that provides the
      probability distribution relative to the walk to another node of the graph.

    v (:obj:`tf.Variable`): The stationary distribution vector. It contains the
      normalized probability to stay in each vertex of the graph. So represents
      the PageRank ranking of the graph.

    writer (:obj:`tf.summary.FileWriter`): This attribute represents a
      TensorFlow's Writer, that is used to obtain stats.

    is_sparse (bool): Use sparse Tensors if it's set to True. Not
      implemented yet.

    iter (:obj:`tf.Tensor`): The operation that will be repeated in each
      iteration of the algorithm.

  """

  def __init__(self, sess: tf.Session, name: str, graph: Graph,
               beta: float, writer: tf.summary.FileWriter = None,
               is_sparse: bool = False) -> None:
    """ Constructor of the class.

    This method initializes the attributes needed to run the Algebraic version
    of PageRank algorithm. It uses the `tf_G.TransitionResetMatrix` as
    transition matrix between vertex.

    Args:

      sess (:obj:`tf.Session`): This attribute represents the session
          that runs the TensorFlow operations.

      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.

      G (:obj:`tf_G.Graph`): The graph on witch it will be calculated the
        algorithm. It will be treated as Directed Weighted Graph.

      beta (float): The reset probability of the random walks, i.e. the
        probability that a user that surfs the graph an decides to jump to
        another vertex not connected to the current.

      v (:obj:`tf.Variable`): The stationary distribution vector. It contains
        the normalized probability to stay in each vertex of the graph. So
        represents the PageRank ranking of the graph.

      writer (:obj:`tf.summary.FileWriter`): This attribute represents a
        TensorFlow's Writer, that is used to obtain stats.

      is_sparse (bool): Use sparse Tensors if it's set to True. Not
        implemented yet.
    """
    T = TransitionResetMatrix(sess, name + "_iter", graph, beta)
    PageRank.__init__(self, sess, name + "_iter", graph, beta, T, writer,
                      is_sparse)
    self.iter = lambda i, a, b=self.T(): tf.matmul(a, b)

  def _pr_convergence_tf(self, convergence: float, topics: List[int] = None,
                         c_criterion=ConvergenceCriterion.ONE) -> tf.Tensor:
    """ Method that implements a iterative version of PageRank until convergence
      rate.

    This method runs the PageRank algorithm in iterative fashion a undetermined
    number of times bounded by the `convergence` rate and the 'c_criterion'
    criterion.

    [TODO describe the algorithm]

    Args:

      convergence (float): A float between 0 and 1 that represents
        the convergence rate that allowed to finish the iterative
        implementations of the algorithm to accept the solution. Default to
        `1.0`.

      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`. Not implemented yet.

      c_criterion (:obj:`function`, optional): The function used to calculate if
        the Convergence Criterion of the iterative implementations is reached.
        Default to `tf_G.ConvergenceCriterion.ONE`.

    Returns:

      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    if topics is not None:
      warnings.warn('Personalized PageRank not implemented yet!')

    self.run_tf(
      self.v.assign(
        tf.while_loop(c_criterion,
                      lambda i, v, v_last, c, n:
                      (i + 1, self.iter(i, v), v, c, n),
                      [0.0, self.v, tf.zeros([1, self.G.n]),
                       convergence,
                       self.G.n_tf], name=self.name + "_while_conv")[
          1]))
    return self.v

  def _pr_steps_tf(self, steps: int, topics: List[int] = None) -> tf.Tensor:
    """ Method that implements a iterative version of PageRank with fixed steps.

    This method runs the PageRank algorithm in iterative fashion a fixed number
    of times bounded by the `steps` parameter.

    [TODO describe the algorithm]

    Args:

      steps (int): A positive integer that sets the number of
        iterations that the iterative implementations will run the algorithm
        until finish. Default to `0`.

      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`. Not implemented yet.

    Returns:

      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    if topics is not None:
      warnings.warn('Personalized PageRank not implemented yet!')

    self.run_tf(
      self.v.assign(
        tf.while_loop(lambda i, v: i < steps,
                      lambda i, v: (i + 1.0, self.iter(i, v)),
                      [0.0, self.v], name=self.name + "_while_steps")[
          1]))
    return self.v

  def _pr_exact_tf(self, topics: List[int] = None) -> None:
    """ This class not implements the exact version of PageRank.

    It generates an exception to notify it to the user.

    Args:

      topics (:obj:`list` of :obj:`int`, optional): A list of integers that
        represent the set of vertex where the random jumps arrives. If this
        parameter is used, the uniform distribution over all vertices of the
        random jumps will be modified to jump only to this vertex set. Default
        to `None`. Not implemented yet.

    Returns:

      (:obj:`tf.Tensor`): A 1-D `tf.Tensor` of [n] shape, where `n` is the
        cardinality of the graph vertex set. It contains the normalized rank of
        vertex `i` at position `i`.

    """
    if topics is not None:
      warnings.warn('Personalized PageRank not implemented yet!')

    raise NotImplementedError(
      str(self.__class__.__name__) + ' not implements exact PageRank')

  def update_edge(self, edge: np.array, change: float) -> None:
    """ The callback to receive notifications about edge changes in the graph.

    This method is called from the Graph when an addition or deletion is
    produced on the edge set. So probably is necessary to recompute the PageRank
    ranking.


    Args:

      edge (:obj:`np.Array`): A 1-D `np.Array` that represents the edge that
        changes in the graph, where `edge[0]` is the source vertex, and
        `edge[1]` the destination vertex.

      change (float): The variation of the edge weight. If the final value is
        0.0 then the edge is removed.

    Returns:

      This method returns nothing.

    """
    self.run_tf(self._pr_convergence_tf(convergence=0.01))
