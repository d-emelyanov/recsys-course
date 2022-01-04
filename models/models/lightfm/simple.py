from datetime import date
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from lightfm import LightFM
from lightfm.data import Dataset
from common.abstract import BaseRecommender
from models.popular.simple import PopularRecommender
from argparse import ArgumentParser


class SimpleLightFM(BaseRecommender):

    lightfm: LightFM = None

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--no_components', type=int, default=10)
        parser.add_argument('--lfm__k', type=int, default=5)
        parser.add_argument('--lfm__n', type=int, default=10)
        parser.add_argument('--lfm__loss', type=str, default='warp')
        parser.add_argument('--c', type=int, default=10)
        parser.add_argument('--watched_pct_lower', type=int, default=10)
        parser.add_argument('--watched_pct_upper', type=int, default=90)
        args, _ = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        no_components=10,
        lfm__k=5,
        lfm__n=10,
        lfm__loss='warp',
        lfm__max_sampled=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lightfm = LightFM(
            no_components=no_components,
            loss=lfm__loss,
            k=lfm__k,
            n=lfm__n,
            max_sampled=lfm__max_sampled
        )

    @property
    def params(self):
        return self.lightfm.get_params()

    def fit(self, df):
        df = df.loc[
            (df.watched_pct >= self.watched_pct_lower)
            & (df.watched_pct <= self.watched_pct_upper)
        ]
        self.data = Dataset()
        self.data.fit(
            users=df[self.user_col].unique().tolist(),
            items=df[self.item_col].unique().tolist()
        )
        interactions, weights = self.data.build_interactions(
            df[[self.user_col, self.item_col]].values.tolist()
        )
        self.lightfm.fit(
            interactions=interactions,
            sample_weight=weights,
        )
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )

    def get_user_representation(self, user_ids):
        uids, _, _ , _ = self.data.mapping()
        if len(list(set(user_ids).difference(set(uids.keys())))) > 0:
            raise ValueError('Some ids are not presented in LightFM train')
        indices = [uids.get(u) for u in user_ids]
        b, e = self.lightfm.get_user_representations()
        return b[indices], e[indices]

    def recommend(self, user_ids, N, with_scores=False):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}

        ub, ue = self.get_user_representation(user_ids)
        ib, ie = self.lightfm.get_item_representations()
        scores = ue.dot(ie.T) + ub.reshape(-1, 1) + ib.reshape(1, -1)
        recs = np.argsort(-scores)

        scores_  = []
        recommendations = []
        for uid_ in tqdm(user_ids):
            seen = [iid.get(x) for x in self.user_seen.loc[uid_]]
            id_recs = [
                k
                for k in recs[uid.get(uid_)]
                if k not in seen
            ][:N]
            recommendations.append([
                iid_reverted[k]
                for k in id_recs
            ])
            scores_.append(scores[uid.get(uid_), id_recs])

        if with_scores:
            return (
                pd.Series(recommendations),
                pd.Series(scores_)
            )
        else:
            return pd.Series(recommendations)


class SimpleWeightedLightFM(SimpleLightFM):

    def fit(self, df):
        df = df.loc[
            (df.watched_pct >= self.watched_pct_lower)
            & (df.watched_pct <= self.watched_pct_upper)
        ]
        self.data = Dataset()
        self.data.fit(
            users=df[self.user_col].unique().tolist(),
            items=df[self.item_col].unique().tolist()
        )
        interactions, weights = self.data.build_interactions(
            df[[self.user_col, self.item_col, 'watched_pct']].values.tolist()
        )
        self.lightfm.fit(
            interactions=interactions,
            sample_weight=weights,
        )
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )