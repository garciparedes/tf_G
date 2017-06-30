import tensorflow as tf

from tf_G import TensorFlowObject


class Edge:
    def __init__(self, src: int, dst: int) -> None:
        self.src = src
        self.dst = dst

    def __call__(self, *args, **kwargs):
        pass
