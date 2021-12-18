import pandas as pd
from unittest import TestCase
from models.hybrid import simple


class TestHybrid(TestCase):

    def setUp(self):
        self.df = pd.DataFrame([
            [1, 1, '2021-01-01'],
            [1, 2, '2021-01-01'],
            [2, 1, '2021-01-01']
        ], columns=['uid', 'iid', 'ts'])

        self.df['ts'] = pd.to_datetime(self.df['ts'])

        self.user_features = pd.DataFrame([
            [1, 1, 100],
            [2, 0, 10],
        ], columns=['uid', 'male', 'income'])

        self.item_features = pd.DataFrame([
            [1, 1],
            [2, 0]
        ], columns=['iid', 'is_action'])

    def test_simple(self):

        rec = simple.SimpleLightFMXgboost(
            no_components=10,
            user_features=['income'],
            item_features=['is_action'],
            item_col='iid',
            user_col='uid',
            date_col='ts'
        )
        rec.add_item_features(self.item_features)
        rec.add_user_features(self.user_features)

        rec.fit(self.df)
        recs = rec.recommend(
            self.df['uid'].unique().tolist(), 10
        )
        print(recs)
        # print(recs)