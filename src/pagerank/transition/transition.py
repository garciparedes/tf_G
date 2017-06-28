from utils.tensorflow_object import TensorFlowObject
from utils.update_edge_notifier import UpdateEdgeNotifier


class Transition(TensorFlowObject, UpdateEdgeNotifier):
    def __init__(self, sess, name, graph):
        TensorFlowObject.__init__(self, sess, name + "_T")
        UpdateEdgeNotifier.__init__(self)

        self.G = graph
        self.G.attach(self)
