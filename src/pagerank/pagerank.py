from src.utils.tensorflow_object import TensorFlowObject


class PageRank(TensorFlowObject):
    def __init__(self, sess, name):
        TensorFlowObject.__init__(self, sess, name)

    def pagerank_vector(self, convergence=1.0, steps=0, personalized=None):
        return self.run(
            self.pagerank_vector_tf(convergence, steps, personalized))

    def pagerank_vector_tf(self, convergence=1.0, steps=0, personalized=None):
        if 0.0 < convergence < 1.0:
            return self._pr_convergence_tf(convergence,
                                           personalized=personalized)
        elif steps > 1:
            return self._pr_steps_tf(steps,
                                     personalized=personalized)
        else:
            return self._pr_exact_tf(personalized=personalized)

    def ranks(self, convergence=1.0, steps=0, personalized=None):
        raise NotImplementedError(
            'subclasses must override ranks()!')

    def _pr_convergence_tf(self, convergence, personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_until_convergence()!')

    def _pr_steps_tf(self, steps, personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_until_steps()!')

    def _pr_exact_tf(self, personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_exact()!')
