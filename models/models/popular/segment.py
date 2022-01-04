import pandas as pd
import logging
from common.abstract import BaseRecommender
from argparse import ArgumentParser
from itertools import product
from recsys_course.preprocess import preprocess_users
from tqdm import tqdm


class SegmentRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--days', type=int)
        parser.add_argument('--segment', type=str, nargs='*')
        parser.add_argument('--fb__min_watched_pct', type=int, default=0)
        parser.add_argument('--fb__total_dur_min', type=int, default=0)
        args, _ = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recommendations = []

    @property
    def params(self):
        return {
            'days': self.days,
            'segment': str(self.segment),
            'fb__min_watched_pct': self.fb__min_watched_pct,
            'fb__total_dur_min': self.fb__total_dur_min
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
        df = df.loc[
            (df['watched_pct'] >= self.fb__min_watched_pct)
            & (df['total_dur'] >= self.fb__total_dur_min)
        ]
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
                recs = df_.loc[df_[self.date_col] > min_date, self.item_col].value_counts()
                self.recommendations[segment] = recs
        min_date = df[self.date_col].max().normalize() - pd.DateOffset(days=self.days)
        self.fallback = df.loc[df[self.date_col] > min_date, self.item_col].value_counts()

    def recommend(self, user_ids, N, with_scores=False):

        def get_recs(x, recommendations, fallback, N):
            r = recommendations.get(tuple(x)).index.values.tolist()
            if r is None:
                r = fallback.index.values.tolist()
            return r[:N]

        def get_scores(x, recommendations, fallback, N):
            r = recommendations.get(tuple(x)).values.tolist()
            if r is None:
                r = fallback.values.tolist()
            return r[:N]

        df = pd.DataFrame({self.user_col: user_ids})
        df = self.get_full_df(df, self.user_col)
        df = preprocess_users(df)
        df['segment'] = df.apply(lambda x: [x[i] for i in self.segment], axis=1)
        df['recs'] = df['segment'].map(
            lambda x: get_recs(
                x,
                self.recommendations,
                self.fallback,
                N
            )
        )
        if with_scores:
            df['scores'] = df['segment'].map(
                lambda x: get_scores(
                    x,
                    self.recommendations,
                    self.fallback,
                    N
                )
            )
            return (
                pd.Series(df['recs'].values),
                pd.Series(df['scores'].values)
            )
        else:
            return pd.Series(df['recs'].values)


class SegmentUnseenRecommender(SegmentRecommender):

    def fit(self, df):
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )
        super().fit(df)

    def recommend(self, user_ids, N):
        recs = []

        def get_recs(x, user_id, recommendations, fallback, seen):
            r = recommendations.get(tuple(x), [])
            if len(r) < N:
                r = fallback
            if user_id in seen.index:
                seen = seen.loc[user_id]
            else:
                seen = []
            recs = [x for x in r if x not in seen]
            return recs[:N]

        df = pd.DataFrame({self.user_col: user_ids})
        df = self.get_full_df(df, self.user_col)
        #df = preprocess_users(df)
        df['segment'] = df.apply(lambda x: [x[i] for i in self.segment], axis=1)
        df['recs'] = df.apply(
            lambda x: get_recs(x['segment'], x[self.user_col], self.recommendations, self.fallback, self.user_seen),
            axis=1
        )

        for uid in tqdm(user_ids):
            seen = set(self.user_seen.loc[uid])
            recs_ = set(self.recommendations).difference(seen)
            recs.append(list(recs_)[:N])

        return df['recs']
