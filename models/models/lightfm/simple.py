from datetime import date
import pandas as pd
import numpy as np
from tqdm import tqdm
from lightfm import LightFM
from lightfm.data import Dataset
from common.abstract import BaseRecommender
from models.popular.simple import PopularRecommender
from argparse import ArgumentParser


class Simple(BaseRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--no_components', type=int, default=10)
        args = parser.parse_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        no_components,
        item_col,
        user_col,
        date_col,
    ):
        self.item_col = item_col
        self.user_col = user_col
        self.date_col = date_col
        self.lightfm = LightFM(
            no_components=no_components
        )

    @property
    def params(self):
        return self.lightfm.get_params()

    def add_item_features(self, data):
        pass

    def add_user_features(self, data):
        pass

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
            sample_weight=weights
        )
        self.fallback = PopularRecommender(
            days=5,
            item_col=self.item_col,
            user_col=self.user_col,
            date_col=self.date_col
        )
        self.fallback.fit(df)

    def recommend(self, user_ids, N):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        recs = []
        for uid_ in tqdm(user_ids):
            if uid_ not in uid.keys():
                recs.append(
                    self.fallback.recommend([uid_], N).tolist()
                )
            else:
                recs.append([
                    iid_reverted[k]
                    for k in np.argsort(-self.lightfm.predict(
                        user_ids=uid.get(uid_),
                        item_ids=list(iid.values())
                    )[:N])
                ])
        return pd.Series(recs)
