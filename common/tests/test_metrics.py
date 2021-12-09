import pandas as pd
import time
import numpy as np
from unittest import TestCase
from common.metrics import (
    precision_at_k,
    precision_at_k_,
    average_precision_at_k,
    map_at_k
)


class TestMetrics(TestCase):

    def generate_random_data(self, n):
        data = pd.DataFrame([
            {
                'uid': i,
                'recs': list(range(10)),
                'interactions': np.random.choice(
                    range(20), np.random.randint(30)
                )
            }
            for i in range(n)
        ])
        return data

    def setUp(self):
        self.data = pd.DataFrame([
            {'uid': 1, 'recs': [2, 3, 4, 5, 1], 'interactions': [1, 10, 9]},
            {'uid': 2, 'recs': [1, 2, 3, 4, 5], 'interactions': [1, 2, 3]},
            {'uid': 3, 'recs': [1, 2, 3, 4, 5], 'interactions': [10, 20, 30, 40, 50]}
        ])

    def test_precision_at_k(self):

        self.data['p@2'] = precision_at_k(2, self.data['recs'], self.data['interactions'])
        self.data['p@10'] = precision_at_k(10, self.data['recs'],  self.data['interactions'])

        self.assertEqual(
            [0/2, 2/2, 0/2],
            self.data['p@2'].tolist()
        )

        self.assertEqual(
            [1/10, 3/10, 0/10],
            self.data['p@10'].tolist()
        )

    def test_average_precision_at_k(self):

        self.assertEqual(
            [0.0, 1.0, 0.0],
            average_precision_at_k(
                3,
                self.data['recs'],
                self.data['interactions']
            ).tolist()
        )

        self.assertEqual(
            [
                precision_at_k_(
                    5,
                    self.data.loc[0, 'recs'],
                    self.data.loc[0, 'interactions']
                ) * 1/1,
                (
                    precision_at_k_(
                        1,
                        self.data.loc[1, 'recs'],
                        self.data.loc[1, 'interactions']
                    )
                    + precision_at_k_(
                        2,
                        self.data.loc[1, 'recs'],
                        self.data.loc[1, 'interactions']
                    )
                    + precision_at_k_(
                        3,
                        self.data.loc[1, 'recs'],
                        self.data.loc[1, 'interactions']
                    )
                ) * 1/3,
                0.0
            ],
            average_precision_at_k(
                5,
                self.data['recs'],
                self.data['interactions']
            ).tolist()
        )

    def test_map_at_k(self):
        self.assertEqual(
            1.0 / 3,
            map_at_k(
                3,
                self.data['recs'],
                self.data['interactions']
            ).tolist()
        )

    def test_map_at_k_execution_time_1000(self):
        data = self.generate_random_data(1000)
        start = time.time()
        map_at_k(20, data['recs'], data['interactions'])
        end = time.time()
        self.assertGreaterEqual(
            5,
            end-start
        )

    def test_map_at_k_execution_time_10000(self):
        data = self.generate_random_data(10000)
        start = time.time()
        map_at_k(20, data['recs'], data['interactions'])
        end = time.time()
        self.assertGreaterEqual(
            5,
            end-start
        )

    def test_map_at_k_execution_time_100000(self):
        data = self.generate_random_data(100000)
        start = time.time()
        map_at_k(20, data['recs'], data['interactions'])
        end = time.time()
        self.assertGreaterEqual(
            15,
            end-start
        )