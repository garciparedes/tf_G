import pandas as pd


class DataSets:
    @staticmethod
    def _get_path():
        return "./../datasets"

    @staticmethod
    def _compose_url(name):
        return DataSets._get_path() + '/' + name + '/' + name + ".csv"

    @staticmethod
    def _compose(name, index_decrement):
        data = pd.read_csv(DataSets._compose_url(name))
        if index_decrement:
            data -= 1
        return data

    @staticmethod
    def followers(index_decrement=True):
        return DataSets._compose('followers', index_decrement)

    @staticmethod
    def wiki_vote(index_decrement=True):
        return DataSets._compose('wiki-Vote', index_decrement)

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
