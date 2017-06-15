import numpy as np


class Utils:
    @staticmethod
    def ranked(x):
        return np.argsort(x, axis=1)

    @staticmethod
    def save_ranks(filename, array, index_increment=True):
        if index_increment:
            array[:, 0] += 1
        np.savetxt(filename, array, fmt='%i,%f',
                   header='vertex_id,page_rank', comments='')
