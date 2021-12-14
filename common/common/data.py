import pandas as pd
import os
import logging

from pandas.core.base import DataError
from .split import TimeRangeSplit


class Dataset:

    def __init__(self, df):
        self.df = df

    @property
    def real(self):
        pass


class DataLoader:

    @classmethod
    def from_folder(cls, files, **kwargs):
        kw = {
            'users': None,
            'items': None,
            'interactions': None
        }
        for fp in os.listdir(files):
            for kwk in kw.keys():
                if kwk in fp:
                    kw[kwk] = os.path.join(files, fp)
        return cls(**kw, **kwargs)

    def __init__(
        self,
        users,
        items,
        interactions,
        user_col,
        item_col,
        date_col
    ):
        self.users = users
        self.items = items
        self.interactions = interactions
        self.interactions[date_col] = pd.to_datetime(
            self.interactions[date_col]
        )
        self.user_col = user_col
        self.item_col = item_col
        self.date_col = date_col

    def get_folds(self, folds, fold_size):
        last_date = self.interactions[self.date_col].max().normalize()
        start_date = last_date - pd.Timedelta(days=folds*fold_size)
        cv = TimeRangeSplit(
            start_date=start_date,
            periods=folds+1,
            freq='W'
        )
        folds_with_stats = list(cv.split(
            self.interactions,
            user_column=self.user_col,
            item_column=self.item_col,
            datetime_column=self.date_col,
            fold_stats=True
        ))

        folds_info_with_stats = pd.DataFrame([info for _, _, info in folds_with_stats])
        logging.info(folds_info_with_stats)

        folds_with_stats = [
            (
                self.interactions.loc[train_idx],
                self.interactions.loc[test_idx],
                info
            )
            for train_idx, test_idx, info in folds_with_stats
        ]

        return folds_with_stats


    def get_real(self, df):
        return (
            df
            .sort_values([self.user_col, self.date_col])
            .groupby(self.user_col)[self.item_col]
            .apply(list)
            .reset_index()
        )

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
