import pandas as pd


class DataSets:
    @staticmethod
    def _get_path():
        return "./datasets"

    @staticmethod
    def _name_to_default_path(name):
        return DataSets._get_path() + '/' + name + '/' + name + ".csv"

    @staticmethod
    def _compose_from_path(path, index_decrement):
        data = pd.read_csv(path)
        if index_decrement:
            data -= 1
        return data.as_matrix()

    @staticmethod
    def _compose_from_name(name, index_decrement):
        return DataSets._compose_from_path(DataSets._name_to_default_path(name),
                                           index_decrement)

    @staticmethod
    def followers(index_decrement=True):
        return DataSets._compose_from_name('followers', index_decrement)

    @staticmethod
    def wiki_vote(index_decrement=True):
        return DataSets._compose_from_name('wiki-Vote', index_decrement)

    @staticmethod
    def generate_from_path(path, index_increment=True):
        return DataSets._compose_from_path(path, index_increment)

    @staticmethod
    def naive():
        """
            url: http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
        """
        return pd.DataFrame([
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 0],
            [3, 0],
            [3, 2]])
