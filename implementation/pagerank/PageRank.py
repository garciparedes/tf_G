from abc import abstractmethod, ABC


class PageRank(ABC):
    def __init__(self, sess):
        self.sess = sess

    @abstractmethod
    def page_rank_vector(self, convergence=None, steps=None):
        pass

    @abstractmethod
    def ranks(self):
        pass
