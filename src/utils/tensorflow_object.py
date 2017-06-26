import warnings

import tensorflow as tf


class TensorFlowObject(object):
    _i = 0

    @staticmethod
    def i():
        TensorFlowObject._i += 1
        return str(TensorFlowObject._i)

    def __init__(self, sess, name, writer=None, is_sparse=False):
        self.sess = sess
        self.name = name
        self.writer = writer
        if is_sparse:
            warnings.warn('TensorFlow not implements Sparse Variables yet!')
        # self.is_sparse = is_sparse
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def run(self, input_to_run):
        r = self.sess.run(input_to_run, options=self.run_options,
                          run_metadata=self.run_metadata)
        if self.writer is not None:
            self.writer.add_run_metadata(self.run_metadata,
                                         TensorFlowObject.i())
        return r
