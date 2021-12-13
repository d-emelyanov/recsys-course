import pandas as pd
from common.abstract import BaseRecommender
from argparse import ArgumentParser


class PopularRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args):
        parser = ArgumentParser()
        parser.add_argument('--days', type=int)
        parser.add_argument('--item_col', type=str, default='item_id')
        parser.add_argument('--user_col', type=str, default='user_id')
        parser.add_argument('--date_col', type=str, default='last_watch_dt')
        args = parser.parse_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        })

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

    def fit(self, df):
        min_date = df[self.date_col].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = df.loc[df[self.date_col] > min_date, self.item_col].value_counts().index.values

    def recommend(self, df, N):
        return pd.DataFrame([
            {
                'user_id': uid,
                'recs': self.recommendations[:N]
            } for uid in df[self.user_col].unique().tolist()
        ])
