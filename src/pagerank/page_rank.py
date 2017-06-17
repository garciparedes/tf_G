from src.tensor_flow_object import TensorFlowObject


class PageRank(TensorFlowObject):
    def __init__(self, sess, name):
        TensorFlowObject.__init__(self, sess, name)

    def page_rank_vector(self, convergence=1.0, steps=0, personalized=None):
        if 0.0 < convergence < 1.0:
            return self._page_rank_until_convergence(convergence, personalized=personalized)
        elif steps > 1:
            return self._page_rank_until_steps(steps,personalized=personalized)
        else:
            return self._page_rank_exact(personalized=personalized)

    def ranks(self, convergence=1.0, steps=0,personalized=None):
        raise NotImplementedError(
            'subclasses must override ranks()!')

    def _page_rank_until_convergence(self, convergence,personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_until_convergence()!')

    def _page_rank_until_steps(self, steps,personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_until_steps()!')

    def _page_rank_exact(self,personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_exact()!')
