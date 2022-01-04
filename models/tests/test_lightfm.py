import pandas as pd
from unittest import TestCase
from models.lightfm import (
    SimpleLightFM,
    SimpleWeightedLightFM
)


class TestLightFM(TestCase):

    def setUp(self) -> None:
        self.df = pd.DataFrame([
            {'uid': 1, 'iid': 1, 'ts': '2021-01-01'},
            {'uid': 1, 'iid': 2, 'ts': '2021-01-01'},
            {'uid': 1, 'iid': 3, 'ts': '2021-01-01'},
            {'uid': 2, 'iid': 5, 'ts': '2021-01-01'},
            {'uid': 2, 'iid': 2, 'ts': '2021-01-01'},
            {'uid': 1, 'iid': 4, 'ts': '2021-01-01'}
        ])
        self.df['ts'] = pd.to_datetime(self.df['ts'])
        self.df['watched_pct'] = 100

    def test_simple(self):
        rec = SimpleLightFM(
            no_components=10,
            item_col='iid',
            user_col='uid',
            date_col='ts',
            watched_pct_min=0,
            watched_pct_max=100
        )
        rec.fit(self.df)
        recs = rec.recommend(user_ids=[1, 2], N=10)
        self.assertEqual(
            [1, 3],
            [len(r) for r in recs.values.tolist()]
        )

    def test_simple_weighted(self):
        rec = SimpleWeightedLightFM(
            no_components=10,
            item_col='iid',
            user_col='uid',
            date_col='ts',
            watched_pct_min=0,
            watched_pct_max=100
        )
        rec.fit(self.df)
        recs = rec.recommend(user_ids=[1, 2], N=10)
        self.assertEqual(
            [1, 3],
            [len(r) for r in recs.values.tolist()]
        )
