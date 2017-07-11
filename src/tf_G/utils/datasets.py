import pandas as pd
import numpy as np


class DataSets:
  """
  DataSets class represents some data sets included in the package.

  The class provides `compose_from_path` method to import personal sets.
  """

  @staticmethod
  def permute_edges(edges_np: np.ndarray) -> np.ndarray:
    """ Method that permutes the rows order of given the input set.

    Args:
      edges_np (:obj:`np.ndarray`): The input data set.

    Returns:
      (:obj:`np.ndarray`): The input data set permuted in rows

    """
    return np.random.permutation(edges_np)

  @staticmethod
  def compose_from_path(path: str, index_decrement: bool) -> np.ndarray:
    """ This method generates a data set from a given path.

    The method obtains the data from the given path, then decrements its values
    if is necessary and permutes the resulting data set.

    The decrement option is offered because of in some cases the data set treats
    the initial node as 1 but many data structures in python are 0-indexed, so
    decrementing the values improves space performance.

    Args:
      path (str): The path of the file of data set csv.
      index_decrement (bool): Decrements all valus by one if True, do nothing
        otherwise.

    Returns:
        (:obj:`np.ndarray`): The data set that represents the Graph.

    """
    data = pd.read_csv(path)
    if index_decrement:
      data -= 1
    return DataSets.permute_edges(data.as_matrix())

  @staticmethod
  def naive_4() -> np.ndarray:
    """ This method returns the naive_4 data set.

    The data set is obtained from Cornell University guide lecture of PageRank
    algorithm.


    This graph contains 4 vertices and 8 edges.

    Url:
      http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html

    Returns:
      (:obj:`np.ndarray`): The data set that represents the Graph.

    """
    return DataSets.permute_edges(np.array([
      [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 0], [3, 0], [3, 2]]))

  @staticmethod
  def naive_6() -> np.ndarray:
    """ This method returns the naive_6 data set.

    The data set is obtained from mathworks study of PageRank algorithm.

    This graph contains 6 vertices and 9 edges.

    Url:
      https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/pagerank.pdf

    Returns:
      (:obj:`np.ndarray`): The data set that represents the Graph.

    """
    return DataSets.permute_edges(np.array([
      [1, 2], [1, 6], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [4, 1],
      [6, 1]]) - 1)
