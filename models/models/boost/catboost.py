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
        return self.catboost.get_all_params()

    @property
    def feature_list(self):
        f = []
        text_features = None
        category_features = None

        if self.text_features:
            text_features = list(range(len(self.text_features)))
            f += self.text_features

        if self.category_features:
            category_features = list(range(len(f), len(f) + len(self.category_features)))
            f += self.category_features

        return f, text_features, category_features

    def fit(self, df):
        logging.info('Training booster')

        logging.info('Getting full df')
        df = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        for tf in self.text_features:
            df[tf] = df[tf].map(lambda x: x.replace(',', ' ').replace('  ', ' '))

        logging.info('Fitting XGBoost')
        (
            feature_list,
            cat_features,
            text_features
        ) = self.feature_list
        X = df[feature_list].values
        y = df['y']

        self.catboost.fit(
            X=X,
            y=y,
            cat_features=cat_features,
            text_features=text_features
        )

    def recommend(self, df, N):
        df = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        for tf in self.text_features:
            df[tf] = df[tf].map(lambda x: x.replace(',', ' ').replace('  ', ' '))
        feature_list, _, _ = self.feature_list
        df['score'] = self.catboost.predict_proba(
            df[feature_list]
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
