from abc import abstractmethod

import numpy as np


class PageRank:
    def __init__(self, sess):
        self.sess = sess

    @abstractmethod
    def page_rank_vector(self):
        pass

    @abstractmethod
    def ranks(self):
        pass
