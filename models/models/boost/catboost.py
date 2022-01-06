from datetime import date
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier
from common.abstract import BaseRecommender
from argparse import ArgumentParser


class CatboostRecommender(BaseRecommender):

    catboost: CatBoostClassifier = None

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--features', type=str, nargs='*')
        parser.add_argument('--category_features', type=str, nargs='*')
        parser.add_argument('--text_features', type=str, nargs='*')
        parser.add_argument('--cb__iterations', type=int, default=20)
        args, _ = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        features = None,
        category_features = None,
        text_features = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        cb_params = {}
        for kw, kv in kwargs.items():
            if kw.startswith('cb__'):
                cb_params[kw[4:]] = kv
        self.catboost = CatBoostClassifier(**cb_params)
        self.features = features
        self.category_features = category_features
        self.text_features = text_features

    @property
    def params(self):
        return self.catboost.get_params()

    @property
    def feature_list(self):
        f = []
        if self.text_features:
            f += self.text_features
        if self.category_features:
            f += self.category_features
        if self.features:
            f += self.features
        return f

    def fit(self, df):
        logging.info('Training booster')

        logging.info('Getting full df')
        df = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        if self.category_features:
            for cf in self.category_features:
                df[cf] = df[cf].fillna('unknown')
        if self.text_features:
            for tf in self.text_features:
                df[tf] = df[tf].fillna('')
                df[tf] = df[tf].map(lambda x: x.replace(',', ' ').replace('  ', ' '))

        logging.info('Fitting Catboost')
        self.catboost.fit(
            X=df[self.feature_list],
            y=df['y'],
            cat_features=self.category_features,
            text_features=self.text_features,
        )

    def recommend(self, df, N):
        df = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        if self.category_features:
            for cf in self.category_features:
                df[cf] = df[cf].fillna('unknown')
        if self.text_features:
            for tf in self.text_features:
                df[tf] = df[tf].fillna('')
                df[tf] = df[tf].map(lambda x: x.replace(',', ' ').replace('  ', ' '))

        df['score'] = self.catboost.predict_proba(
            df[self.feature_list]
        )[:, 1]

        df = (
            df
            .groupby(self.user_col)
            .apply(lambda x: pd.Series({
                self.item_col: (
                    x.sort_values('score', ascending=False)[self.item_col]
                    .tolist()[:N]
                )
            }))
        )

        return df
