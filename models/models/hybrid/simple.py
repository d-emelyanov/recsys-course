from datetime import date
import pandas as pd
import logging
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm
from lightfm import LightFM
from lightfm.data import Dataset
from common.abstract import BaseRecommender
from models.popular.simple import PopularRecommender
from models.lightfm.simple import SimpleLightFM
from argparse import ArgumentParser


class SimpleLightFMXGBoost(BaseRecommender):

    lightfm: SimpleLightFM = None
    xgboost: XGBClassifier = None
    fallback: PopularRecommender = None

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--no_components', type=int, default=10)
        parser.add_argument('--user_features',  type=str, nargs='*')
        parser.add_argument('--item_features', type=str, nargs='*')
        args = parser.parse_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        no_components,
        user_features,
        item_features,
        item_col,
        user_col,
        date_col,
    ):
        self.item_col = item_col
        self.user_col = user_col
        self.date_col = date_col
        self.user_features_col = user_features
        self.item_features_col = item_features
        self.lightfm = SimpleLightFM(
            no_components=no_components,
            item_col=item_col,
            user_col=user_col,
            date_col=date_col
        )
        self.xgboost = XGBClassifier()
        self.fallback = PopularRecommender(
            days=5,
            item_col=item_col,
            user_col=user_col,
            date_col=date_col
        )

    @property
    def params(self):
        return {
            **{f'xgb_{k}': v for k, v in self.xgboost.get_params().items()}
            **{f'lfm_{k}': v for k, v in self.lightfm.params.items()}
        }

    @property
    def features(self):
        return self.user_features_col + self.item_features_col

    def fit(self, df):

        logging.info('Training lightfm model')
        self.lightfm.fit(df)

        logging.info('Training booster')

        unused = self.get_full_df(
            data=self.unused,
            user_col=self.user_col,
            item_col=self.item_col
        )
        unused['y'] = 0

        data = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        data['y'] =  1
        data = data.drop(self.date_col, axis=1)
        data = pd.concat([unused, data]).reset_index(drop=True)

        self.xgboost.fit(data[self.features], data['y'])

        logging.info('Training fallback')
        self.fallback.fit(df)

    def recommend(self, user_ids, N):

        uid, _, _, _ = self.lightfm.data.mapping()
        seen_users = set(list(uid.keys()))
        cold_users = set(user_ids) ^ seen_users

        df = []

        logging.info('Get fallback recommendations')
        df_ = pd.DataFrame({self.user_col: list(cold_users)})
        df_[self.item_col] = self.fallback.recommend(
            df_[self.user_col].tolist(),
            N
        )
        df.append(df_)

        logging.info('Get lightfm recommendations')
        df_ = pd.DataFrame({self.user_col: list(seen_users)})
        df_[self.item_col] = self.lightfm.recommend(seen_users, N=100)

        logging.info('Get xgboost reccommendations')
        df_ = df_.explode(self.item_col)
        df_ = self.get_full_df(
            data=df_,
            user_col=self.user_col,
            item_col=self.item_col
        )
        df_['score'] = self.xgboost.predict_proba(
            df_[self.features]
        )[:, 1]
        df_ = (
            df_
            .groupby(self.user_col)
            .apply(lambda x: pd.Series({
                self.item_col: (
                    x.sort_values('score', ascending=False)[self.item_col]
                    .tolist()[:N]
                )
            }))
            .reset_index()
        )
        df.append(df_)

        df = (
            pd
            .concat(df)
            .reset_index(drop=True)
            .set_index(self.user_col)
        )
        return df.loc[user_ids, self.item_col]
