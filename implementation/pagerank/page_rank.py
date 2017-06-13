from tensor_flow_object import TensorFlowObject


class PageRank(TensorFlowObject):
    def __init__(self, sess, name):
        TensorFlowObject.__init__(self, sess, name)

    def page_rank_vector(self, convergence=None, steps=None):
        if convergence is not None:
            return self.page_rank_until_convergence(convergence)
        elif steps > 0:
            return self.page_rank_until_steps(steps)
        else:
            raise ValueError("'convergence' or 'steps' must be assigned!")

    def page_rank_until_convergence(self, convergence):
        raise NotImplementedError(
            'subclasses must override page_rank_until_convergence()!')

    def page_rank_until_steps(self, steps):
        raise NotImplementedError(
            'subclasses must override page_rank_until_steps()!')

    def ranks(self, convergence=None, steps=None):
        raise NotImplementedError(
            'subclasses must override ranks()!')
