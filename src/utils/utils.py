import numpy as np


class Utils:
    @staticmethod
    def ranked(x: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.argsort(x, axis=axis)

    @staticmethod
    def save_ranks(filename: str, array: np.ndarray,
                   index_increment: bool = True) -> None:
        if index_increment:
            array[:, 0] += 1
        np.savetxt(filename, array, fmt='%i,%f',
                   header='vertex_id,page_rank', comments='')
