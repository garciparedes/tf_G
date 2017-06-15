class TensorFlowObject(object):
    def __init__(self, sess, name, writer=None):
        self.sess = sess
        self.name = name
        self.writer = writer

    def run(self, input_to_run):
        return self.sess.run(input_to_run)
