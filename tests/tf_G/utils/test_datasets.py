import tfgraph


def test_data_sets_naive_4():
  assert tfgraph.DataSets.naive_4().shape == (8, 2)


def test_data_sets_naive_6():
  assert tfgraph.DataSets.naive_6().shape == (9, 2)


def test_data_sets_compose():
  assert tfgraph.DataSets.compose_from_path("./datasets/wiki-Vote/wiki-Vote.csv",
                                         True).shape == (65499, 2)
