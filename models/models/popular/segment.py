import pandas as pd
from common.abstract import BaseRecommender
from argparse import ArgumentParser


class PopularRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--days', type=int)
        args = parser.parse_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        days,
        item_col,
        user_col,
        date_col,
    ):
        self.days = days
        self.item_col = item_col
        self.user_col = user_col
        self.date_col = date_col
        self.recommendations = []

    @property
    def params(self):
        return {
            'days': self.days
        }

    def add_item_features(self, data):
        pass

    def add_user_features(self, data):
        pass

    def fit(self, df):
        min_date = df[self.date_col].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = df.loc[df[self.date_col] > min_date, self.item_col].value_counts().index.values.tolist()

    def recommend(self, user_ids, N):
        return pd.Series([self.recommendations[:N] for _ in range(len(user_ids))])
