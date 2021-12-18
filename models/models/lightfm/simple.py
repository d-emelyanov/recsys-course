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

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--no_components', type=int, default=10)
        args = parser.parse_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)


    lightfm: LightFM = None
    fallback: PopularRecommender = None


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
        self.fallback = PopularRecommender(
            days=5,
            item_col=item_col,
            user_col=user_col,
            date_col=date_col
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
            sample_weight=weights
        )
        self.fallback.fit(df)

    def recommend(self, user_ids, N):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        recs = []
        for uid_ in tqdm(user_ids):
            if uid_ not in uid.keys():
                recs.append(
                    self.fallback.recommend([uid_], N).tolist()[0]
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

    def recommend_(self, user_ids, N):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        uid_reverted = {v: k for k, v in uid.items()}
        recs = {}
        cold = []
        warm = []
        logging.info('Cold split')
        for uid_ in user_ids:
            if uid_ in uid.keys():
                warm.append(uid.get(uid_))
            else:
                cold.append(uid_)
        logging.info(f'{len(cold) / len(user_ids):.1%}% of cold users')

        logging.info('Getting cold recs')
        cold_recs = dict(zip(
            cold,
            self.fallback.recommend(cold, N).tolist()
        ))

        logging.info('Getting warm recs')
        item_ids_ = np.concatenate(
            [list(iid.values()) for _ in tqdm(range(len(warm)))],
            dtype=np.int32
        )
        # todo: slow thing here
        user_ids_ = np.concatenate(
            [[x for _ in range(len(iid))] for x in tqdm(warm)],
            dtype=np.int32
        )
        logging.info('Getting predictions')
        predictions = self.lightfm.predict(
            user_ids=user_ids_,
            item_ids=item_ids_,
            num_threads=3
        )
        warm_recs = {
            uid_reverted.get(r[0]): [iid_reverted.get(iid_) for iid_ in r[1]]
            for r in tqdm(zip(
                warm,
                np.argsort(
                    -predictions.reshape(len(warm), -1)
                )[:, :N].tolist()
            ))
        }

        logging.info('Generating predictions')
        recs = {
            **cold_recs,
            **warm_recs
        }

        return pd.Series([recs.get(u) for u in tqdm(user_ids)])
