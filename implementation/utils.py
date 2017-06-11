import numpy as np


class Utils:
    @staticmethod
    def ranked(x):
        return np.argsort(x, axis=0)

    @staticmethod
    def save(filename, array):
        np.savetxt(filename, array, fmt='%i', delimiter=',')
