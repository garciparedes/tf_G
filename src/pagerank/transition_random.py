import tensorflow as tf

from src.utils.tensorflow_object import TensorFlowObject


class TransitionRandom(TensorFlowObject):
    def __init__(self, sess, name, T, writer=None):
        TensorFlowObject.__init__(self, sess, name, writer=writer)
        self.T = T

    def get(self, vertex):
        pass
