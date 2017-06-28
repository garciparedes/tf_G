import pandas as pd
import numpy as np


class DataSets:
    @staticmethod
    def _get_path() -> str:
        return "./../datasets"

    @staticmethod
    def _name_to_default_path(name: str) -> str:
        return DataSets._get_path() + '/' + name + '/' + name + ".csv"

    @staticmethod
    def _permute_edges(edges_np: np.ndarray) -> np.ndarray:
        return np.random.permutation(edges_np)

    @staticmethod
    def _compose_from_path(path: str, index_decrement: bool) -> np.ndarray:
        data = pd.read_csv(path)
        if index_decrement:
            data -= 1
        return DataSets._permute_edges(data.as_matrix())

    @staticmethod
    def _compose_from_name(name: str, index_decrement: bool) -> np.ndarray:
        return DataSets._compose_from_path(DataSets._name_to_default_path(name),
                                           index_decrement)

    @staticmethod
    def followers(index_decrement: bool = True) -> np.ndarray:
        return DataSets._compose_from_name('followers', index_decrement)

    @staticmethod
    def wiki_vote(index_decrement: bool = True) -> np.ndarray:
        return DataSets._compose_from_name('wiki-Vote', index_decrement)

    @staticmethod
    def generate_from_path(path: str, index_increment=True) -> np.ndarray:
        return DataSets._compose_from_path(path, index_increment)

    @staticmethod
    def naive_4() -> np.ndarray:
        """
            url: http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
        """
        return DataSets._permute_edges(np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 0],
            [3, 0],
            [3, 2]]))

    @staticmethod
    def naive_6() -> np.ndarray:
        """
            url: https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/pagerank.pdf
        """
        return DataSets._permute_edges(np.array([
            [1, 2],
            [1, 6],
            [2, 3],
            [2, 4],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 1],
            [6, 1]]) - 1)
