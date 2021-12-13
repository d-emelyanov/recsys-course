import pandas as pd
import os


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

    @property
    def train(self):
        pass

    @property
    def test(self):
        pass

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
