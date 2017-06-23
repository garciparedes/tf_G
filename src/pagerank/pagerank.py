import warnings

from src.utils.tensorflow_object import TensorFlowObject
from src.utils.vector_convergence import VectorConvergenceCriterion


class PageRank(TensorFlowObject):
    def __init__(self, sess, name):
        TensorFlowObject.__init__(self, sess, name)

    def error_vector_compare_tf(self, other_pr):
        raise NotImplementedError(
            'subclasses must override compare()!')

    def error_vector_compare_np(self, other_pr):
        return self.run(self.error_vector_compare_tf(other_pr))

    def pagerank_vector_np(self, convergence=1.0, steps=0, personalized=None):
        return self.run(
            self.pagerank_vector_tf(convergence, steps, personalized))

    def pagerank_vector_tf(self, convergence=1.0, steps=0, personalized=None,
                           convergence_criterion=VectorConvergenceCriterion.ONE):
        if 0.0 < convergence < 1.0:
            return self._pr_convergence_tf(convergence,
                                           personalized=personalized,
                                           convergence_criterion=convergence_criterion)
        elif steps > 0:
            return self._pr_steps_tf(steps,
                                     personalized=personalized)
        else:
            return self._pr_exact_tf(personalized=personalized)

    def ranks(self, convergence=1.0, steps=0, personalized=None):
        raise NotImplementedError(
            'subclasses must override ranks()!')

    def _pr_convergence_tf(self, convergence, personalized,
                           convergence_criterion):
        raise NotImplementedError(
            'subclasses must override page_rank_until_convergence()!')

    def _pr_steps_tf(self, steps, personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_until_steps()!')

    def _pr_exact_tf(self, personalized):
        raise NotImplementedError(
            'subclasses must override page_rank_exact()!')

    def update_edge(self, edge, change):
        warnings.warn('PageRank auto-update not implemented yet!')
