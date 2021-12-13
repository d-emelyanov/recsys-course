import pandas as pd
import os

from pandas.core.base import DataError


class Dataset:

    def __init__(self, df):
        self.df = df

    @property
    def real(self):
        pass


class DataLoader:

    @classmethod
    def from_folder(cls, files):
        kw = {
            'users': None,
            'items': None,
            'interactions': None
        }
        for fp in os.listdir(files):
            for kwk in kw.keys():
                if kwk in fp:
                    kw[kwk] = os.path.join(files, fp)
        return cls(**kw)

    def __init__(
        self,
        users,
        items,
        interactions
    ):
        self.users = users
        self.items = items
        self.interactions = interactions
        self.interactions['last_watch_dt'] = pd.to_datetime(
            self.interactions['last_watch_dt']
        )

    def train_test_split(self, k=0.7):
        dates_ = (
            self.interactions['last_watch_dt']
            .unique()
        )
        n = int(len(dates_) * k)

        self.train = self.interactions.loc[
            (self.interactions['last_watch_dt'] >= dates_[:n].min())
            & (self.interactions['last_watch_dt'] <= dates_[:n].max())
        ]

        self.test = self.interactions.loc[
            (self.interactions['last_watch_dt'] > dates_[n:].min())
            & (self.interactions['last_watch_dt'] <= dates_[n:].max())
        ]

    def get_real(self, df):
        return (
            df
            .sort_values(['user_id', 'last_watch_dt'])
            .groupby('user_id')['item_id']
            .apply(list)
            .reset_index()['item_id']
        )

    @property
    def train_real(self):
        return self.get_real(self.train)

    @property
    def test_real(self):
        return self.get_real(self.test)

    @property
    def users(self):
        return self.__users

    @property
    def items(self):
        return self.__items

    @property
    def interactions(self):
        return self.__interactions

    @users.setter
    def users(self, var):
        if var:
            self.__users = pd.read_csv(var)
        else:
            self.__users = None

    @items.setter
    def items(self, var):
        if var:
            self.__items = pd.read_csv(var)
        else:
            self.__items = None

    @interactions.setter
    def interactions(self, var):
        if var:
            self.__interactions = pd.read_csv(var)
        else:
            self.__interactions = None
