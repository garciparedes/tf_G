import tensorflow as tf


class TransitionMatrix:
    def __init__(self, graph):
        self.G = graph

    @property
    def get(self):
        return tf.div(self.G.A_tf, self.G.E_o_degrees)
