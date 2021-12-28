import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser
from .simple import SimpleLightFM
from lightfm import LightFM
from lightfm.data import Dataset
from tqdm import tqdm


class FeaturedLightFM(SimpleLightFM):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--no_components', type=int, default=10)
        parser.add_argument('--lfm__k', type=int, default=5)
        parser.add_argument('--lfm__n', type=int, default=10)
        parser.add_argument('--lfm__loss', type=str, default='warp')
        parser.add_argument('--lfm__max_sampled', type=int, default=10)
        parser.add_argument('--user_features_col', type=str, nargs='*', default=None)
        parser.add_argument('--item_features_col', type=str, nargs='*', default=None)
        parser.add_argument('--preprocess_array_split', type=str, nargs='*', default=None)
        args, _ = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        user_features_col,
        item_features_col,
        preprocess_array_split,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.user_features_col = user_features_col
        self.item_features_col = item_features_col
        self.preprocess_array_split = preprocess_array_split

    @property
    def params(self):
        return {
            **super().params,
            'user_features_col': self.user_features_col,
            'item_features_col': self.item_features_col,
            'preprocess_array_split': self.preprocess_array_split
        }

    @property
    def use_item_features(self):
        return self.item_features_col is not None

    @property
    def use_user_features(self):
        return self.user_features_col is not None

    def get_unique_values(self, df, cols):
        vals = []
        for col in cols:
            if self.preprocess_array_split and col in self.preprocess_array_split:
                df_ = df.explode(col)
                vals += df_[col].unique().tolist()
            else:
                vals += df[col].unique().tolist()
        return list(set(vals))

    def split_str_to_arr(self, df, delim=','):
        if self.preprocess_array_split:
            for col in self.preprocess_array_split:
                if col in df.columns:
                    df[col] = df[col].map(lambda x: x.split(delim))
        return df

    def get_features(self, id_col, df, cols, filter_id=None):

        def comb(x, cols):
            ret = []
            for c in cols:
                if isinstance(x[c], list):
                    ret += x[c]
                else:
                    ret += [x[c]]
            return ret

        if filter_id:
            df = df.loc[df[id_col].isin(filter_id)]

        df['comb'] = df.apply(lambda x: comb(x, cols), axis=1)
        return df[[id_col, 'comb']].values

    def fit(self, df):
        self.data = Dataset()

        logging.info('Preparing user features')
        if self.use_user_features:
            self.user_features = self.split_str_to_arr(self.user_features)
            ufs = {
                'user_features': self.get_unique_values(
                    self.user_features,
                    self.user_features_col
                )
            }
        else:
            ufs = {}

        logging.info('Preparing item features')
        if self.use_item_features:
            self.item_features = self.split_str_to_arr(self.item_features)
            ifs = {
                'item_features': self.get_unique_values(
                    self.item_features,
                    self.item_features_col
                )
            }
        else:
            ifs = {}

        logging.info('Fitting lightfm dataset')
        self.data.fit(
            users=df[self.user_col].unique().tolist(),
            items=df[self.item_col].unique().tolist(),
            **ufs, **ifs
        )

        logging.info('Building interactions')
        interactions, weights = self.data.build_interactions(
            df[[self.user_col, self.item_col]].values.tolist()
        )

        if self.use_user_features:
            logging.info('Building user features')
            user_features = {
                'user_features': self.data.build_user_features(
                    self.get_features(
                        self.user_col,
                        self.user_features,
                        self.user_features_col,
                        filter_id=df[self.user_col].unique().tolist()
                    )
                )
            }
        else:
            user_features = {}

        if self.use_item_features:
            logging.info('Building item features')
            item_features = {
                'item_features': self.data.build_item_features(
                    self.get_features(
                        self.item_col,
                        self.item_features,
                        self.item_features_col,
                        filter_id=df[self.item_col].unique().tolist()
                    )
                )
            }
        else:
            item_features = {}

        logging.info('Fitting lightfm model')
        self.lightfm.fit(
            interactions=interactions,
            sample_weight=weights,
            **user_features, **item_features
        )
        self.user_seen = (
            df
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )

    def recommend(self, user_ids, N):
        uid, _, iid, _ = self.data.mapping()
        iid_reverted = {v: k for k, v in iid.items()}
        uid_reverted = {v: k for k, v in uid.items()}
        if self.use_user_features:
            user_features = {
                'user_features': self.data.build_user_features(
                    self.get_features(
                        self.user_col,
                        self.user_features,
                        self.user_features_col,
                        filter_id=uid.keys()
                    )
                )
            }
        else:
            user_features = {}

        if self.use_item_features:
            item_features = {
                'item_features': self.data.build_item_features(
                    self.get_features(
                        self.item_col,
                        self.item_features,
                        self.item_features_col,
                        filter_id=iid.keys()
                    )
                )
            }
        else:
            item_features = {}

        recs = []
        for uid_ in tqdm(user_ids):
            if uid_ in self.user_seen.index:
                seen = set([iid.get(x) for x in self.user_seen.loc[uid_]])
                not_seen = list(set(iid.values()).difference(seen))
            else:
                not_seen = list(iid.values())
            recs.append([
                iid_reverted[not_seen[k]]
                for k in np.argsort(-self.lightfm.predict(
                    user_ids=uid.get(uid_),
                    item_ids=not_seen,
                    **user_features, **item_features
                ))[:N]
            ])
        return pd.Series(recs)


class WeightFeaturedLightFM(FeaturedLightFM):

    @classmethod
    def from_args(cls, args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--no_components', type=int, default=10)
        parser.add_argument('--notseen_watched_lower', type=int, default=10)
        parser.add_argument('--notseen_watched_upper', type=int, default=90)
        parser.add_argument('--lfm__k', type=int, default=5)
        parser.add_argument('--lfm__n', type=int, default=10)
        parser.add_argument('--lfm__loss', type=str, default='warp')
        parser.add_argument('--lfm__max_sampled', type=int, default=10)
        parser.add_argument('--user_features_col', type=str, nargs='*', default=None)
        parser.add_argument('--item_features_col', type=str, nargs='*', default=None)
        parser.add_argument('--preprocess_array_split', type=str, nargs='*', default=None)
        args, _ = parser.parse_known_args(args)
        return cls(**{
            k: v
            for k, v in vars(args).items()
        }, **kwargs)

    def __init__(
        self,
        notseen_watched_lower=10,
        notseen_watched_upper=90,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.notseen_watched_lower = notseen_watched_lower
        self.notseen_watched_upper = notseen_watched_upper

    @property
    def params(self):
        return {
            **super().params,
            'notseen_watched_lower': self.notseen_watched_lower,
            'notseen_watched_upper': self.notseen_watched_upper,
        }

    def fit(self, df):
        self.data = Dataset()

        logging.info('Preparing user features')
        if self.use_user_features:
            self.user_features = self.split_str_to_arr(self.user_features)
            ufs = {
                'user_features': self.get_unique_values(
                    self.user_features,
                    self.user_features_col
                )
            }
        else:
            ufs = {}

        logging.info('Preparing item features')
        if self.use_item_features:
            self.item_features = self.split_str_to_arr(self.item_features)
            ifs = {
                'item_features': self.get_unique_values(
                    self.item_features,
                    self.item_features_col
                )
            }
        else:
            ifs = {}

        logging.info('Fitting lightfm dataset')
        self.data.fit(
            users=df[self.user_col].unique().tolist(),
            items=df[self.item_col].unique().tolist(),
            **ufs, **ifs
        )

        logging.info('Building interactions')
        # df['w'] = 1 / df['watched_pct'] * 100
        interactions, weights = self.data.build_interactions(
            df[[self.user_col, self.item_col, 'watched_pct']].values.tolist()
        )

        if self.use_user_features:
            logging.info('Building user features')
            user_features = {
                'user_features': self.data.build_user_features(
                    self.get_features(
                        self.user_col,
                        self.user_features,
                        self.user_features_col,
                        filter_id=df[self.user_col].unique().tolist()
                    )
                )
            }
        else:
            user_features = {}

        if self.use_item_features:
            logging.info('Building item features')
            item_features = {
                'item_features': self.data.build_item_features(
                    self.get_features(
                        self.item_col,
                        self.item_features,
                        self.item_features_col,
                        filter_id=df[self.item_col].unique().tolist()
                    )
                )
            }
        else:
            item_features = {}

        logging.info('Fitting lightfm model')
        self.lightfm.fit(
            interactions=interactions,
            sample_weight=weights,
            **user_features, **item_features
        )
        self.user_seen = (
            df.loc[
                (df.watched_pct <= self.notseen_watched_lower)
                | (df.watched_pct >= self.notseen_watched_upper)
            ]
            .groupby(self.user_col)[self.item_col]
            .apply(list)
        )