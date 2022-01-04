from datetime import date
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
from common.abstract import BaseRecommender
from argparse import ArgumentParser
from common.preprocess import (
    preprocess_items,
    preprocess_users,
    onehot
)


class XGBoostRecommender(BaseRecommender):

    xgboost: XGBClassifier = None

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--user_features_col', type=str, nargs='*')
        parser.add_argument('--item_features_col', type=str, nargs='*')
        parser.add_argument('--category_features', type=str, nargs='*')
        args, _ = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        user_features_col,
        item_features_col,
        category_features,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.xgboost = XGBClassifier()
        self.user_features_col = user_features_col
        self.item_features_col = item_features_col
        self.category_features = category_features

    @property
    def params(self):
        return self.lightfm.get_params()

    @property
    def features(self):
        return self.user_features_col + self.item_features_col

    def fit(self, df):
        logging.info('Training booster')

        logging.info('Getting unused full df')
        unused = self.get_full_df(
            data=self.unused.loc[
                self.unused[self.user_col].isin(df[self.user_col].unique().tolist())
            ],
            user_col=self.user_col,
            item_col=self.item_col
        )
        unused['y'] = 0

        logging.info('Getting interactions full df')
        data = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        data['y'] =  1
        data = data.drop(self.date_col, axis=1)
        data = pd.concat([unused, data]).reset_index(drop=True)

        logging.info('Checking onehot')
        feature_list = []
        for feature_ in self.features:
            if feature_ in self.category_features:
                logging.info(f'Onehot for {feature_}')
                data = onehot(data, [feature_], True)
                feature_list += [x for x in data.columns if x.startswith(f'{feature_}_')]
            else:
                feature_list.append(feature_)

        logging.info('Fitting XGBoost')
        self.xgboost.fit(data[feature_list], data['y'])

    def recommend(self, df, N):
        df = self.get_full_df(
            data=df,
            user_col=self.user_col,
            item_col=self.item_col
        )
        feature_list = []
        for feature_ in self.features:
            if feature_ in self.category_features:
                logging.info(f'Onehot for {feature_}')
                df = onehot(df, [feature_], True)
                feature_list += [x for x in df.columns if x.startswith(f'{feature_}_')]
            else:
                feature_list.append(feature_)
        df['score'] = self.xgboost.predict_proba(
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
            .reset_index()
        )

        return df[[self.user_col, self.item_col]]
