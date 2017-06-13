from abc import abstractmethod, ABC


class PageRank(ABC):
    def __init__(self, sess):
        self.sess = sess

    def page_rank_vector(self, convergence=None, steps=None):
        if convergence is not None:
            return self.page_rank_until_convergence(convergence)
        elif steps > 0:
            return self.page_rank_until_steps(steps)
        else:
            raise ValueError("'convergence' or 'steps' must be assigned")

    @abstractmethod
    def page_rank_until_convergence(self, convergence):
        pass

    @abstractmethod
    def page_rank_until_steps(self, steps):
        pass

    @abstractmethod
    def ranks(self, convergence=None, steps=None):
        pass
