import pandas as pd
import logging
from common.abstract import BaseRecommender
from argparse import ArgumentParser
from itertools import product
from tqdm import tqdm


class SegmentRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--days', type=int)
        parser.add_argument('--watched_pct_min', type=int, default=0)
        parser.add_argument('--segment', type=str, nargs='*')
        args = parser.parse_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        days,
        watched_pct_min,
        segment,
        item_col,
        user_col,
        date_col,
    ):
        self.days = days
        self.watched_pct_min = watched_pct_min
        self.item_col = item_col
        self.user_col = user_col
        self.date_col = date_col
        self.segment = segment
        self.recommendations = []

    @property
    def params(self):
        return {
            'days': self.days,
            'watched_pct_min': self.watched_pct_min,
            'segment': str(self.segment)
        }

    def get_segments(self, df):
        vals = []
        for segment in self.segment:
            vals.append(df[segment].unique().tolist())
        return product(*vals)

    def get_full_df(self, data, user_col):
        data = pd.merge(
            left=data,
            right=self.user_features,
            on=[user_col],
            how='left'
        )
        return data

    def fit(self, df):
        df = df.loc[df['watched_pct'] >= self.watched_pct_min]
        df = self.get_full_df(df, self.user_col)
        self.recommendations = {}
        for segment in tqdm(self.get_segments(df)):
            logging.info(f'Fitting model for segment: {dict(zip(self.segment, segment))}')
            mask = None
            for k, v in zip(self.segment, segment):
                if mask is None:
                    mask = (df[k] == v)
                else:
                    mask = mask & (df[k] == v)
            df_ = df.loc[mask]
            if df_.shape[0] > 0:
                min_date = df_[self.date_col].max().normalize() - pd.DateOffset(days=self.days)
                recs = df_.loc[df_[self.date_col] > min_date, self.item_col].value_counts().index.values.tolist()
                self.recommendations[segment] = recs
        min_date = df[self.date_col].max().normalize() - pd.DateOffset(days=self.days)
        self.fallback = df.loc[df[self.date_col] > min_date, self.item_col].value_counts().index.values.tolist()

    def recommend(self, user_ids, N):

        def get_recs(x, recommendations, fallback, N):
            r = recommendations.get(tuple(x), [])
            if len(r) < N:
                r = fallback
            return r[:N]

        df = pd.DataFrame({self.user_col: user_ids})
        df = self.get_full_df(df, self.user_col)
        df['segment'] = df.apply(lambda x: [x[i] for i in self.segment], axis=1)
        df['recs'] = df['segment'].map(lambda x: get_recs(x, self.recommendations, self.fallback, N))
        return df['recs']
