import tf_G


def test_data_sets_naive_4():
  assert tf_G.DataSets.naive_4().shape == (8, 2)


def test_data_sets_naive_6():
  assert tf_G.DataSets.naive_6().shape == (9, 2)


def test_data_sets_compose():
  assert tf_G.DataSets.compose_from_path("./datasets/wiki-Vote/wiki-Vote.csv",
                                         True).shape == (65499, 2)
