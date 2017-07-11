import numpy as np
import tf_G
import os


def test_ranking():
  np.testing.assert_array_equal(
    tf_G.Utils.ranked(np.array([[1, 3, 2, 4]])),
    np.array([[0, 2, 1, 3]]))


def test_save():
  file_name = 'proof.csv'
  init = np.array([[0, 0.5], [1, 0.5]])
  tf_G.Utils.save_ranks(file_name, init)

  end = np.genfromtxt(file_name, delimiter=',', skip_header=1)

  np.testing.assert_array_equal(init, end)
  os.remove(file_name)
