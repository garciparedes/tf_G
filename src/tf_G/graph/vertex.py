import tensorflow as tf
import numpy as np
from tf_G import TensorFlowObject


class Edge:
    def __init__(self, in_edges: np.ndarray, out_edges: np.ndarray) -> None:
        self.in_edges = in_edges
        self.out_edges = out_edges

    def __call__(self, *args, **kwargs):
        pass


