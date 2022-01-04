import pandas as pd
from common.abstract import BaseRecommender
from argparse import ArgumentParser

from tqdm.std import tqdm


class PopularRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--days', type=int, default=5)
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
        params = {}
        if hasattr(self, 'fb__min_watched_pct'):
            params = {
                **params,
                **{'fb__min_watched_pct': self.fb__min_watched_pct}
            }
        if hasattr(self, 'fb__min_watched_pct'):
            params = {
                **params,
                **{'fb__total_dur_min': self.fb__total_dur_min}
            }
        return {
            'days': self.days,
            **params
        }

    def add_item_features(self, data):
        return None

    def add_user_features(self, data):
        return None

    def fit(self, df):
        mask = None
        if hasattr(self, 'fb__min_watched_pct'):
            mask = (df['watched_pct'] >= self.fb__min_watched_pct)
        if hasattr(self, 'fb__total_dur_min'):
            mask = mask & (df['total_dur'] >= self.fb__total_dur_min)
        if mask is not None:
            df = df.loc[mask]
        min_date = df[self.date_col].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = df.loc[df[self.date_col] > min_date, self.item_col].value_counts()

    def recommend(self, user_ids, N, with_scores=False):
        recs = self.recommendations.index.values.tolist()
        if with_scores:
            scores = self.recommendations.values.tolist()
            return (
                pd.Series([recs[:N] for _ in range(len(user_ids))]),
                pd.Series([scores[:N] for _ in range(len(user_ids))]),
            )
        else:
            return pd.Series([recs[:N] for _ in range(len(user_ids))])


class PopularUnseenRecommmender(PopularRecommender):

    def fit(self, df):
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )
        super().fit(df)

    def recommend(self, user_ids, N):
        recs = []
        for uid in tqdm(user_ids):
            if uid in self.user_seen.index:
                seen = self.user_seen.loc[uid]
            else:
                seen = []
            recs_ = [x for x in self.recommendations if x not in seen]
            recs.append(recs_[:N])
        return pd.Series(recs)