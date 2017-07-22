import numpy as np


class Utils:
  """ Utils class of the tfgraph package.

  This class contains static methods that will be used by another classes
  around the package.

  """

  @staticmethod
  def ranked(x: np.ndarray) -> np.ndarray:
    """ This method sorts the array indices given by its values.

    It can be used to generate a ranking based on the values of an array in
    which the value represents the score and the index the object identifier.

    Args:
      x (:obj:`np.ndarray`): A 2-D `np.ndarray` to rank the results by rows.

    Returns:
      (:obj:`np.ndarray`): An array containing the indices of the input array
        sorted in decremental order.

    """
    return np.argsort(x, axis=1)

  @staticmethod
  def save_ranks(filename: str, array: np.ndarray,
                 index_increment: bool = True) -> None:
    """ This method will save the input array in the filesystem.

    The method creates a file in the filesystem with name `filename` and puts
    the array content inside it.

    This method provides the `index_increment` that increments the first column
    by one if True. The reason of this option is that the method is created to
    store results of graph operations, and in some cases the graph vertices is
    represented as 0-indexed and anothers as 1-indexed.

    Args:
      filename (str): The name of the file that will be created.
      array (:obj:`np.ndarray`): The array that contains the data that will be
        saved.
      index_increment (bool, optional): Increments the first column of the array
        input if True, do nothing otherwise.

    Returns:
      This method returns nothing.

    """
    if index_increment:
      array[:, 0] += 1
    np.savetxt(filename, array, fmt='%i,%f',
               header='vertex_id,page_rank', comments='')
