import logging
import pandas as pd
from common.abstract import BaseRecommender
from argparse import ArgumentParser
from models.loader import load_model
from sklearn.preprocessing import StandardScaler


class TwoStageRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--models', nargs='*', type=str, required=True)
        parser.add_argument('--models_w', nargs='*', type=float, required=True)
        parser.add_argument('--models_n', type=int, required=True)
        parser.add_argument('--final_model', type=str, required=True)
        parser.add_argument('--final_model_max_sample', type=int)
        args, params = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, params=params, **kwargs)

    def __init__(
        self,
        models,
        models_w,
        models_n,
        final_model,
        final_model_max_sample,
        params,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.models = []
        for model, model_w in zip(models, models_w):
            self.models.append({
                'model': load_model(model, params, **kwargs),
                'weight': model_w,
                'name': model
            })
        self.final_model_max_sample = final_model_max_sample
        self.final_model = load_model(final_model, params, **kwargs)
        self.models_n = models_n

    @property
    def params(self):
        params = {}
        for model in self.models:
            params = {
                **params,
                **{
                    f"{model['name']}__{k}": v
                    for k, v in model['model'].params.items()
                }
            }
        return {
            **params,
            **self.final_model.params
        }

    def add_user_features(self, data):
        for model in self.models:
            model['model'].add_user_features(data)
        self.final_model.add_user_features(data)

    def add_item_features(self, data):
        for model in self.models:
            model['model'].add_item_features(data)
        self.final_model.add_item_features(data)

    def fit(self, df):
        logging.info('Fitting models')

        user_ids = df[self.user_col].unique().tolist()
        recs = None
        for i, model in enumerate(self.models):
            logging.info('Fitting model: ' + model['name'])
            model['model'].fit(df)
            recs_, scores_ = model['model'].recommend(
                user_ids,
                N=self.models_n,
                with_scores=True
            )

            df_ = pd.DataFrame({self.user_col: user_ids})
            df_[self.item_col] = recs_
            df_['score'] = scores_
            df_ = df_.explode([self.item_col, 'score'])
            df_['score_scaled'] = (
                StandardScaler()
                .fit_transform(df_['score'].values.reshape(-1, 1))
            ) * model['weight']
            if recs is None:
                recs = df_.copy()
            else:
                recs = pd.merge(
                    left=(
                        pd.concat([recs, df_])
                        .reset_index(drop=True)
                        .groupby([
                            self.user_col,
                            self.item_col
                        ])['score_scaled']
                        .sum()
                        .reset_index()
                    ),
                    right=(
                        df_[[self.user_col, self.item_col, 'score']]
                        .rename(columns={'score': f'score_{i}'})
                    ),
                    on=[
                        self.user_col,
                        self.item_col
                    ],
                    how='left'
                ).fillna(0.0)

        logging.info('Prepare booster data')
        recs = pd.merge(
            left=recs,
            right=df,
            on=[self.user_col, self.item_col],
            how='left'
        )
        recs['y'] = recs[self.date_col].map(lambda x: 0  if pd.isna(x) else 1)

        logging.info('Fitting booster')
        self.final_model.fit(recs)

    def recommend(self, user_ids, N):
        logging.info('Start recommmending')

        recs = None
        for i, model in enumerate(self.models):
            logging.info('Getting recs from model: ' + model['name'])
            recs_, scores_ = model['model'].recommend(
                user_ids,
                N=self.models_n,
                with_scores=True
            )

            df_ = pd.DataFrame({self.user_col: user_ids})
            df_[self.item_col] = recs_
            df_['score'] = scores_
            df_ = df_.explode([self.item_col, 'score'])
            df_['score_scaled'] = (
                StandardScaler()
                .fit_transform(df_['score'].values.reshape(-1, 1))
            ) * model['weight']

            if recs is None:
                recs = df_.copy()
            else:
                recs = pd.merge(
                    left=(
                        pd.concat([recs, df_])
                        .reset_index(drop=True)
                        .groupby([
                            self.user_col,
                            self.item_col
                        ])['score_scaled']
                        .sum()
                        .reset_index()
                    ),
                    right=(
                        df_[[self.user_col, self.item_col, 'score']]
                        .rename(columns={'score': f'score_{i}'})
                    ),
                    on=[
                        self.user_col,
                        self.item_col
                    ],
                    how='left'
                ).fillna(0.0)

        logging.info('Predicting booster')
        final_recs = self.final_model.recommend(recs, N)
        return final_recs.loc[user_ids, self.item_col].tolist()