
class TensorFlowObject(object):

    def __init__(self,sess, name):
        self.sess = sess
        self.name = name