import pandas as pd
import logging
import numpy as np
from common.abstract import BaseRecommender
from argparse import ArgumentParser
from models.loader import load_model
from tqdm import tqdm


class CombineRecommender(BaseRecommender):

    user_seen = []

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--models', nargs='*', type=str, required=True)
        parser.add_argument('--models_n', type=int, required=True)
        args, params = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, params=params, **kwargs)

    def __init__(
        self,
        models,
        models_n,
        params,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.models_n = models_n
        self.models = []
        for model in models:
            model_ = load_model(model, params, **kwargs)
            self.models.append((model, model_))

    @property
    def params(self):
        params = {}
        for i, (model_name, model) in enumerate(self.models):
            params = {
                **params,
                **{f'{i}__{model_name}__{k}': v for k, v in model.params.items()}
            }
        return params

    def add_user_features(self, data):
        super().add_user_features(data)
        for _, model in self.models:
            model.add_user_features(data)

    def add_item_features(self, data):
        self.item_features = data
        for _, model in self.models:
            model.add_item_features(data)

    def add_unused(self, data):
        self.unused = data
        for _, model in self.models:
            model.add_unused(data)

    def fit(self, df):
        for (model_name, model) in self.models:
            logging.info(f'Training {model_name}..')
            model.fit(df)

    def recommend(self, user_ids, N):

        def explode(df, icol):
            df[icol] = df[icol].map(
                lambda x: list(enumerate(x))
            )
            df = df.explode(icol)
            df[['rank', icol]] = df[icol].apply(pd.Series)
            return df

        def score(x, N):
            return np.power(
                np.prod(x),
                1 / N
            )

        df = []
        for model_name, model in self.models:
            logging.info(f'Getting recs from {model_name}')
            df_ = pd.DataFrame({self.user_col: user_ids})
            df_[self.item_col] = model.recommend(
                df_[self.user_col].tolist(), self.models_n
            )
            df_ = explode(df_, self.item_col)
            df.append(df_)

        logging.info('Generating Final Recs')
        recs = (
            pd
            .concat(df)
            .groupby([self.user_col, self.item_col])['rank']
            .apply(lambda x: score(x, self.models_n))
            .reset_index()
            .groupby(self.user_col)
            .apply(lambda x: x.sort_values('rank')[self.item_ccol].tolist()[:N])
            .loc[user_ids]
            .values
        )

        return pd.Series(recs)


class CombineUnseenRecommender(CombineRecommender):

    def fit(self, df):
        super().fit(df)
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )


class TwoStepRecommender(CombineRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--models', nargs='*', type=str, required=True)
        parser.add_argument('--models_n', nargs='*', type=int, required=True)
        parser.add_argument('--final_model', type=str, required=True)
        args, params = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, params=params, **kwargs)

    def __init__(
        self,
        final_model,
        params,
        **kwargs
    ):
        super().__init__(params=params, **kwargs)
        self.final_model = load_model(final_model, params, **kwargs)

    @property
    def params(self):
        return {
            **super().params()
            **self.final_model.params
        }

    def add_user_features(self, data):
        super().add_user_features(data)
        self.final_model.add_user_features(data)

    def add_item_features(self, data):
        super().add_item_features(data)
        self.final_model.add_item_features(data)

    def add_unused(self, data):
        super().add_unused(data)
        self.final_model.add_unused(data)

    def fit(self, df):
        super().fit(df)
        self.final_model.fit(df)

    def recommend(self, user_ids, N):
        df = []
        for (model_name, model), n in zip(self.models, self.models_n):
            logging.info(f'Getting recs from {model_name}')
            df_ = pd.DataFrame({self.user_col: user_ids})
            df_[self.item_col] = model.recommend(
                df_[self.user_col].tolist(), n
            )
            df_ = df_.explode(self.item_col)
            df.append(df_)
        df = pd.concat(df).drop_duplicates().reset_index(drop=True)
        recs = (
            self
            .final_model
            .recommend(df, N)
            .set_index(self.user_col)
        )
        return recs.loc[user_ids, self.item_col].tolist()
