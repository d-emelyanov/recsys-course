from models.lightfm.featured import WeightFeaturedLightFM
import pandas as pd
import os
from datetime import date
from unittest import TestCase
from common.data import DataLoader
from common.tuning import Optimizer
from trainer import Trainer
from models.popular import (
    SegmentRecommender,
    PopularRecommender
)
from models.hybrid import (
    TwoStageRecommender
)
from models.lightfm  import (
    SimpleLightFM,
    SimpleWeightedLightFM
)
from const import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, 'tmp')


class TestTrain(TestCase):

    def _check(**check_values):
        def decorator(func):
            def wrapper(self):
                metrics = func(self)
                self.assertEqual(
                    len(metrics.keys()), 1
                )
                if check_values is not None:
                    for k, v in check_values.items():
                        self.assertEqual(
                            metrics[k], v
                        )
            return wrapper
        return decorator

    def tearDown(self):
        for f in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, f))
        os.removedirs(TMP_DIR)

    def setUp(self):
        if os.path.exists(TMP_DIR) is False:
            os.makedirs(TMP_DIR)

        pd.DataFrame([
            (1, 'age_25_34', 'income_10_20', 'm', 1),
            (2, 'age_25_34', 'income_20_30', 'zh', 0),
            (3, 'age_25_34', 'income_20_30', 'm', 1)
        ], columns=[
            'user_id',
            'age',
            'income',
            'sex',
            'kids_flg'
        ]).to_csv(os.path.join(TMP_DIR, 'users.csv'), index=None)

        pd.DataFrame([
            (1, 'ct1', 't1', 't1', 2010, 'g1, g2', 'c1', 1, 6, 's1', 'd1', 'a1', 'des1', 'kw1, kw2', 'inf_2010', 0),
            (2, 'ct2', 't2', 't2', 2020, 'g2', 'c1', 0, 18, 's2', 'd2', 'a2', 'des2', 'kw2, kw3', '2020_inf', 1),
            (3, 'ct1', 't3', 't3', 2011, 'g1, g2', 'c2', -1, 12, 's3', 'd3', 'a3', 'des3', 'kw1, kw3', '2010_2020', 1),
            (4, 'ct2', 't4', 't4', 2021, 'g2, g3', 'c3', 0,  4, 's4', 'd4', 'a4', 'des4', 'kw3', '2020_inf', 0),
            (5, 'ct1', 't5', 't5', 2009, 'g4', 'c4', 1, 5, 's5', 'd5', 'a5', 'des5', 'kw5', 'inf_2010', 1)
        ], columns=[
            'item_id',
            'content_type',
            'title',
            'title_orig',
            'release_year',
            'genres',
            'countries',
            'for_kids',
            'age_rating',
            'studios',
            'directors',
            'actors',
            'description',
            'keywords',
            'release_year_cat',
            'for_kids_rating'
        ]).to_csv(os.path.join(TMP_DIR, 'items.csv'), index=None)

        pd.DataFrame([
            (1, 1, '2021-12-06', 4250, 72),
            (1, 2, '2021-12-07', 4300, 90),
            (2, 1, '2021-12-08', 4250, 72),
            (3, 3, '2021-12-08', 4100, 25),
            (3, 4, '2021-12-08', 5000, 30),
            (1, 5, '2021-12-14', 3200, 72),
            (1, 3, '2021-12-18', 4100, 10),
            (2, 4, '2021-12-19', 5000, 30)
        ], columns=[
            'user_id',
            'item_id',
            'last_watch_dt',
            'total_dur',
            'watched_pct'
        ]).to_csv(os.path.join(TMP_DIR, 'interactions.csv'), index=None)

        self.data = DataLoader.from_folder(
            TMP_DIR,
            watched_pct_min=0,
            user_col=USER_COL,
            item_col=ITEM_COL,
            date_col=DATE_COL
        )

    @_check(map10=0.85)
    def test_popular_recommender(self):
        params = ['--days', '10', '--watched_pct_min', '0']
        trainer = Trainer(
            params=params,
            data=self.data,
            rec_class=PopularRecommender,
            fb_class=None,
            test_size=0.3,
            n_recs=10,
        )
        metrics, params, (rec, _) = trainer.train()
        return metrics

    # @_check(map10=1.0)
    # def test_popular_recommender_folds(self):
    #     params = ['--days', '10', '--watched_pct_min', '0']
    #     trainer = Trainer(
    #         params=params,
    #         data=self.data,
    #         rec_class=PopularRecommender,
    #         fb_class=None,
    #         folds=1,
    #         n_recs=10,
    #     )

    #     metrics, params, (_, _) = trainer.train()
    #     return metrics

    # @_check(map10=0.85)
    # def test_popular_recommender__optuna(self):
    #     params = [
    #         '--days__type', 'int',
    #         '--days__low', '10',
    #         '--days__high', '10',
    #         '--watched_pct_min__type', 'int',
    #         '--watched_pct_min__low', '0',
    #         '--watched_pct_min__high', '0'
    #     ]
    #     trainer = Trainer(
    #         params=params,
    #         data=self.data,
    #         rec_class=PopularRecommender,
    #         fb_class=None,
    #         test_size=0.3,
    #         n_recs=10,
    #     )
    #     optimizer = Optimizer.from_args(params)
    #     optimizer.optimize(
    #         trials=5,
    #         trainer=trainer,
    #     )
    #     metrics = optimizer.best_metrics
    #     params = optimizer.best_params
    #     return metrics

    # @_check(map10=1.0)
    # def test_popular_recommender__folds__optuna(self):
    #     params = [
    #         '--days__type', 'int',
    #         '--days__low', '10',
    #         '--days__high', '10',
    #         '--watched_pct_min__type', 'int',
    #         '--watched_pct_min__low', '0',
    #         '--watched_pct_min__high', '0'
    #     ]
    #     trainer = Trainer(
    #         params=params,
    #         data=self.data,
    #         rec_class=PopularRecommender,
    #         fb_class=None,
    #         folds=1,
    #         n_recs=10,
    #     )
    #     optimizer = Optimizer.from_args(params)
    #     optimizer.optimize(
    #         trials=5,
    #         trainer=trainer,
    #     )
    #     metrics = optimizer.best_metrics
    #     params = optimizer.best_params
    #     return metrics

    @_check()
    def test_simple_lightfm(self):
        params = [
            '--days', '10',
            '--watched_pct_min', '0',
            '--no_components', '10'
        ]
        trainer = Trainer(
            params=params,
            data=self.data,
            rec_class=SimpleLightFM,
            fb_class=PopularRecommender,
            test_size=0.3,
            n_recs=10,
        )
        metrics, params, (rec, _) = trainer.train()
        return metrics

    @_check()
    def test_simple_weighted_lightfm(self):
        params = [
            '--days', '10',
            '--watched_pct_lower', '0',
            '--no_components', '10'
        ]
        trainer = Trainer(
            params=params,
            data=self.data,
            rec_class=SimpleWeightedLightFM,
            fb_class=PopularRecommender,
            test_size=0.3,
            n_recs=10,
        )
        metrics, params, (rec, _) = trainer.train()
        return metrics

    #@_check()
    def test_two_stage(self):
        params = [
            '--days', '10',
            '--watched_pct_min', '0',
            '--no_components', '10',
            '--models', 'lightfm.SimpleLightFM',
            '--models_n', '100',
            '--models_w', '1.0',
            '--final_model_max_sample', '10',
            '--final_model', 'boost.CatboostRecommender',
            '--features', 'score_0',
            '--exclude_seen', '0',
            #'--category_features', 'content_type', 'age', 'sex',
            #'--text_features', 'keywords',
            '--watched_pct_lower', '0',
            '--no_components', '10',
            #'--segment', 'age'
        ]
        trainer = Trainer(
            params=params,
            data=self.data,
            rec_class=TwoStageRecommender,
            fb_class=PopularRecommender,
            test_size=0.3,
            n_recs=10,
        )
        metrics, params, (rec, _) = trainer.train()

        return metrics