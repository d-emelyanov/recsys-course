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
        parser.add_argument('--lfm__max_sampled', type=int, default=10)
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

    def recommend(self, user_ids, N):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        recs = []

        for uid_ in tqdm(user_ids):
            seen = set([iid.get(x) for x in self.user_seen.loc[uid_]])
            not_seen = list(set(iid.values()).difference(seen))
            recs.append([
                iid_reverted[not_seen[k]]
                for k in np.argsort(-self.lightfm.predict(
                    user_ids=uid.get(uid_),
                    item_ids=not_seen
                ))[:N]
            ])
        return pd.Series(recs)
