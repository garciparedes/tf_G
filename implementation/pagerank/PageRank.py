from abc import abstractmethod, ABC


class PageRank(ABC):
    def __init__(self, sess):
        self.sess = sess

    @abstractmethod
    def page_rank_vector(self, steps):
        pass

    @abstractmethod
    def ranks(self):
        pass
