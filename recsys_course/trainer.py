import pandas as pd
import numpy as np
import importlib
from common.metrics import map_at_k
from common.data import DataLoader
from argparse import ArgumentError
from typing import Union, Any


def get_model_class(model):
    if model is None:
        return None
    rec_module = '.'.join(model.split('.')[:-1])
    module = importlib.import_module(f'models.{rec_module}')
    rec_class_name = model.split('.')[-1]
    rec_class = getattr(module, rec_class_name)
    return rec_class


class Trainer:

    def __init__(
        self,
        params: Union[dict, list],
        rec_class,
        data: DataLoader,
        n_recs: int,
        folds: int = None,
        test_size: float = None,
        fb_class = None
    ):
        self.rec_class = rec_class
        self.fb_class = fb_class
        self.params = params
        self.data = data
        self.n_recs = n_recs
        self.folds = folds
        self.test_size = test_size

    @property
    def train(self):
        if self.test_size:
            return self.train_whole
        elif self.folds:
            return self.train_folds
        else:
            raise ValueError('You need to specify either --test_size or --folds')

    @property
    def has_fallback(self):
        return self.fb_class is not None

    def update_params(self, params):
        self.params = params

    def get_params(self, rec, fallback):
        params = {}
        if self.has_fallback:
            params = {
                **params,
                **{f'fallback_{k}': v for k, v in fallback.params.items()}
            }
        params = {
            **params,
            **{k: v for k, v in rec.params.items()}
        }
        return params

    def evaluate(self, user_ids: list, rec):
        df = pd.DataFrame({self.data.user_col: user_ids})
        df = pd.merge(
            left=df,
            right=self.data.get_real_items(df[self.data.user_col].unique().tolist()),
            on=[self.data.user_col]
        )
        if df.shape[0] == 0:
            raise ValueError('Trainer.evaluate :: no data to evaluate')
        df['recs'] = rec.recommend(
            df[self.data.user_col].tolist(),
            N=self.n_recs
        )

        metrics = {
            f'map{self.n_recs}': map_at_k(
                k=self.n_recs,
                recs=df['recs'],
                real=df['real']
            )
        }
        return metrics

    def initiate_models(self):
        fallback = None
        if isinstance(self.params, list):
            rec = self.rec_class.from_args(
                self.params,
                user_col=self.data.user_col,
                item_col=self.data.item_col,
                date_col=self.data.date_col
            )
            if self.has_fallback:
                fallback = self.fb_class.from_args(
                    self.params,
                    user_col=self.data.user_col,
                    item_col=self.data.item_col,
                    date_col=self.data.date_col
                )
            else:
                fallback = None
        elif isinstance(self.params, dict):
            rec = self.rec_class(
                **self.params,
                user_col=self.data.user_col,
                item_col=self.data.item_col,
                date_col=self.data.date_col
            )
            if self.has_fallback:
                fallback = self.fb_class(
                    **self.params,
                    user_col=self.data.user_col,
                    item_col=self.data.item_col,
                    date_col=self.data.date_col
                )
        else:
            raise TypeError('Params has wrong type')

        if rec:
            if self.data.has_items:
                rec.add_item_features(self.data.items)
            if self.data.has_users:
                rec.add_user_features(self.data.users)
            if self.data.has_unused:
                rec.add_unused(self.data.unused)

        return rec, fallback

    def train_step(self, rec, fallback, train, test):
        metrics = {}
        rec.fit(train)
        if self.has_fallback:
            test_cold = test.loc[
                ~test[self.data.user_col].isin(
                    train[self.data.user_col].unique().tolist()
                )
            ]
            test_warm = test.loc[
                test[self.data.user_col].isin(
                    train[self.data.user_col].unique().tolist()
                )
            ]
            if test_cold.shape[0] > 0:
                metrics_cold = self.evaluate(
                    user_ids=test_cold[self.data.user_col].unique().tolist(),
                    rec=fallback,
                )
                metrics = {
                    **metrics,
                    **{f'fallback_{k}': v for k, v in metrics_cold.items()}
                }
        else:
            test_warm = test.copy()

        metrics_warm = self.evaluate(
            user_ids=test_warm[self.data.user_col].unique().tolist(),
            rec=rec
        )

        metrics = {
            **metrics,
            **{k: v for k, v in metrics_warm.items()}
        }
        return metrics

    def train_whole(self):
        params = {}
        rec, fallback = self.initiate_models()
        train, test = self.data.get_train_test(self.test_size)
        metrics = self.train_step(rec, fallback, train, test)
        params = self.get_params(rec, fallback)
        return metrics, params, (rec, fallback)

    def train_folds(self):
        metrics = {}
        params = {}
        rec, fallback = self.initiate_models()
        for train, test, info in self.data.get_folds(self.folds):
            metrics_ = self.train_step(rec, fallback, train, test)
            for k, v in metrics_.items():
                if k in metrics:
                    metrics[k].append(v)
                else:
                    metrics[k] = [v]

        metrics = {
            k: np.mean(v) for k, v in metrics.items()
        }

        params = self.get_params(rec, fallback)

        return metrics, params, (None, None)
