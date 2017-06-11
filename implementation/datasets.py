import pandas as pd


class DataSets:
    @staticmethod
    def _get_path():
        return "./../datasets"

    @staticmethod
    def _compose_url(name):
        return DataSets._get_path() + '/' + name + '/' + name + ".csv"

    @staticmethod
    def _compose(name):
        return pd.read_csv(DataSets._compose_url(name))

    @staticmethod
    def followers():
        return DataSets._compose('followers')

    @staticmethod
    def wiki_vote():
        return DataSets._compose('wiki-Vote')
