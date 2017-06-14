from tensor_flow_object import TensorFlowObject


class PageRank(TensorFlowObject):
    def __init__(self, sess, name):
        TensorFlowObject.__init__(self, sess, name)

    def page_rank_vector(self, convergence=1.0, steps=0):
        if 0.0 < convergence < 1.0:
            return self._page_rank_until_convergence(convergence)
        elif steps > 1:
            return self._page_rank_until_steps(steps)
        else:
            return self._page_rank_exact()

    def ranks(self, convergence=1.0, steps=0):
        raise NotImplementedError(
            'subclasses must override ranks()!')

    def _page_rank_until_convergence(self, convergence):
        raise NotImplementedError(
            'subclasses must override page_rank_until_convergence()!')

    def _page_rank_until_steps(self, steps):
        raise NotImplementedError(
            'subclasses must override page_rank_until_steps()!')

    def _page_rank_exact(self):
        raise NotImplementedError(
            'subclasses must override page_rank_exact()!')
