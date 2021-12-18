import pandas as pd
from unittest import TestCase
from models.lightfm import (
    simple
)


class TestLightFM(TestCase):

    def setUp(self) -> None:
        self.df = pd.DataFrame([
            {'uid': 1, 'iid': 1, 'ts': '2021-01-01'},
            {'uid': 1, 'iid': 2, 'ts': '2021-01-01'},
            {'uid': 1, 'iid': 3, 'ts': '2021-01-01'},
            {'uid': 2, 'iid': 2, 'ts': '2021-01-02'}
        ])
        self.df['ts'] = pd.to_datetime(self.df['ts'])

    def test_simple(self):
        rec = simple.Simple(
            no_components=10,
            item_col='iid',
            user_col='uid',
            date_col='ts'
        )
        rec.fit(self.df)
        recs = rec.recommend(user_ids=[1, 2, 3], N=10)
        self.assertEqual(
            [3, 3, 3],
            [len(r) for r in recs.values.tolist()]
        )
