import pandas as pd
from unittest import TestCase
from models import lightfm


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
        self.assertEqual(1, 1)
