import pandas as pd
import numpy as np


class DataSets:
  """
  DataSets class represents some data sets included in the package.

  Many of this sets are imported from SNAP project of Stanford University.
  The class also provides `generate_from_path` method to import personal sets.
  """

  @staticmethod
  def _get_path() -> str:
    """ Private method to get the path of provided data sets.

    Returns:
      str: The relative path that points to data sets directory.

    """
    return "./../datasets"

  @staticmethod
  def _name_to_default_path(name: str) -> str:
    """ Private method that returns the path of a set from it's name.

    Args:
      name (str): The name of data set.

    Returns:
      str: The relative path that points to the csv file that contains the data
        set.

    """
    return DataSets._get_path() + '/' + name + '/' + name + ".csv"

  @staticmethod
  def _permute_edges(edges_np: np.ndarray) -> np.ndarray:
    """ Private method that permutes the rows order of given the input set.

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
    return DataSets._permute_edges(data.as_matrix())

  @staticmethod
  def _compose_from_name(name: str, index_decrement: bool) -> np.ndarray:
    """ Private method that composes a data set from its name.

    This method uses `_name_to_default_path` to obtain the path and generates
    the data set using `_compose_from_path`.

    Args:
      name (str): The name of the data set.
      index_decrement (bool): Decrements all valus by one if True, do nothing
        otherwise.

    Returns:
      (:obj:`np.ndarray`): The data set that represents the Graph.

    """
    return DataSets.compose_from_path(DataSets._name_to_default_path(name),
                                      index_decrement)

  @staticmethod
  def followers(index_decrement: bool = True) -> np.ndarray:
    """ This method returns the followers data set.

    The data set is obtained from a example of GraphX, a graph library developed
    on the top of Apache Spark.

    This graph contains 7 vertex and 8 edges.

    Args:
      index_decrement (bool): Decrements all valus by one if True, do nothing
        otherwise.

    Returns:
      (:obj:`np.ndarray`): The data set that represents the followers Graph.

    """
    return DataSets._compose_from_name('followers', index_decrement)

  @staticmethod
  def wiki_vote(index_decrement: bool = True) -> np.ndarray:
    """ This method returns the wiki-Vote data set.

    The data set is obtained from the Stanford's University SNAP project, that
    is based on the study of massive graphs.

    This graph contains 7115 vertices and 103689 edges.

    Url:
      https://snap.stanford.edu/data/wiki-Vote.html

    Args:
      index_decrement (bool): Decrements all valus by one if True, do nothing
        otherwise.

    Returns:
      (:obj:`np.ndarray`): The data set that represents the wiki_vote Graph.

    """
    return DataSets._compose_from_name('wiki-Vote', index_decrement)

  @staticmethod
  def p2p_gnutella08(index_decrement: bool = False) -> np.ndarray:
    """ This method returns the p2p-gnutella08 data set.

    The data set is obtained from the Stanford's University SNAP project, that
    is based on the study of massive graphs.

    This graph contains 6301 vertices and 20777 edges.

    Url:
      https://snap.stanford.edu/data/p2p-Gnutella08.html

    Args:
      index_decrement (bool): Decrements all valus by one if True, do nothing
        otherwise.

    Returns:
      (:obj:`np.ndarray`): The data set that represents the p2p_gnutella08 Graph.

    """
    return DataSets._compose_from_name('p2p-gnutella08', index_decrement)

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
    return DataSets._permute_edges(np.array([
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
    return DataSets._permute_edges(np.array([
      [1, 2], [1, 6], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [4, 1],
      [6, 1]]) - 1)
