import tensorflow as tf

from utils.tensorflow_object import TensorFlowObject
from utils.update_edge_notifier import UpdateEdgeNotifier
from src.graph.graph import Graph


class Transition(TensorFlowObject, UpdateEdgeNotifier):
    def __init__(self, sess: tf.Session, name: str, graph: Graph) -> None:
        TensorFlowObject.__init__(self, sess, name + "_T")
        UpdateEdgeNotifier.__init__(self)

        self.G = graph
        self.G.attach(self)
