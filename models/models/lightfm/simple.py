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
        parser.add_argument('--exclude_seen', type=int, default=0)
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
        exclude_seen=0,
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
        self.exclude_seen = exclude_seen

    @property
    def params(self):
        return self.lightfm.get_params()

    def fit(self, df):
        users_ = df[self.user_col].unique().tolist()
        items_ = df[self.item_col].unique().tolist()
        df = df.loc[
            (df.watched_pct >= self.watched_pct_lower)
            & (df.watched_pct <= self.watched_pct_upper)
        ]
        self.data = Dataset()
        self.data.fit(
            users=users_,
            items=items_
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

    def recommend_seen(self, user_ids, N, with_scores=False):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        recs = []
        scores = []
        for uid_ in tqdm(user_ids):
            scores_ = self.lightfm.predict(
                user_ids=uid.get(uid_),
                item_ids=list(iid.values())
            )
            if with_scores:
                scores.append(np.sort(-scores_)[:N].tolist())
            recs.append([
                iid_reverted[k]
                for k in np.argsort(-scores_)[:N]
            ])
        if with_scores:
            return (
                pd.Series(recs),
                pd.Series(scores)
            )
        else:
            return pd.Series(recs)

    def recommend_unseen(self, user_ids, N, with_scores=False):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        recs = []
        scores = []
        for uid_ in tqdm(user_ids):
            seen = [iid.get(x) for x in self.user_seen.loc[uid_]]
            scores_ = self.lightfm.predict(
                user_ids=uid.get(uid_),
                item_ids=list(iid.values())
            )
            recs_ = [x for x in np.argsort(-scores_) if x not in seen][:N]
            if with_scores:
                scores.append(scores_[recs_].tolist())
            recs.append([
                iid_reverted[k]
                for k in recs_
            ])
        if with_scores:
            return (
                pd.Series(recs),
                pd.Series(scores)
            )
        else:
            return pd.Series(recs)

    def recommend(self, user_ids, N, with_scores=False):
        if self.exclude_seen == 1:
            return self.recommend_unseen(user_ids, N, with_scores)
        else:
            return self.recommend_seen(user_ids, N, with_scores)

    # def recommend__(self, user_ids, N, with_scores=False):
    #     uid, _, iid, _ = self.data.mapping()
    #     iid_reverted = {v: k for k, v in iid.items()}

    #     logging.info('---get reppresentations')
    #     ub, ue = self.get_user_representation(user_ids)
    #     ib, ie = self.lightfm.get_item_representations()
    #     logging.info('---calculate scores')
    #     scores = ue.dot(ie.T) + ub.reshape(-1, 1) + ib.reshape(1, -1)
    #     logging.info('---sorting scores')
    #     recs = np.argsort(-scores)

    #     scores_  = []
    #     recommendations = []
    #     for uid_ in tqdm(user_ids):
    #         id_recs = [
    #             k
    #             for k in recs[uid.get(uid_)]
    #             if k not in seen
    #         ][:N]
    #         recommendations.append([
    #             iid_reverted[k]
    #             for k in id_recs
    #         ])
    #         scores_.append(scores[uid.get(uid_), id_recs])

    #     if with_scores:
    #         return (
    #             pd.Series(recommendations),
    #             pd.Series(scores_)
    #         )
    #     else:
    #         return pd.Series(recommendations)


class SimpleWeightedLightFM(SimpleLightFM):

    def fit(self, df):
        users_ = df[self.user_col].unique().tolist()
        items_ = df[self.item_col].unique().tolist()
        df = df.loc[
            (df.watched_pct >= self.watched_pct_lower)
            & (df.watched_pct <= self.watched_pct_upper)
        ]
        self.data = Dataset()
        logging.info('--fitting dataset')
        self.data.fit(
            users=users_,
            items=items_
        )
        logging.info('--building interactions')
        interactions, weights = self.data.build_interactions(
            df[[self.user_col, self.item_col, 'watched_pct']].values.tolist()
        )
        logging.info('--fitting lightfm')
        self.lightfm.fit(
            interactions=interactions,
            sample_weight=weights,
        )
        logging.info('--getting users seen')
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )