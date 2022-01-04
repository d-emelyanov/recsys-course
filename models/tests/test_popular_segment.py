import pandas as pd
from models.popular import SegmentRecommender
from unittest import TestCase


class TestSegmentRecommender(TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame([
            # male
            (1, 1, '2021-01-01'),
            (1, 2, '2021-01-01'),
            (2, 1, '2021-01-01'),
            # female
            (4, 1, '2021-01-01'),
            (4, 2, '2021-01-01'),
            (5, 2, '2021-01-01')
        ], columns=['uid', 'iid', 'ts'])
        self.data['watched_pct'] = 100
        self.data['total_dur'] = 1000
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.user_features = pd.DataFrame([
            (1, 'm', '10-15', '1000+', 1),
            (2, 'm', '10-15', '1000+', 1),
            (3, 'm', '10-15', '1000+', 1),
            (4, 'f', '10-15', '1000+', 1),
            (5, 'f', '10-15', '1000+', 1)
        ], columns=['uid', 'sex', 'age', 'income', 'kids_flg'])
        return super().setUp()

    def test_fit(self):

        model = SegmentRecommender(
            days=1,
            watched_pct_min=0,
            segment=['sex'],
            item_col='iid',
            user_col='uid',
            date_col='ts',
            fb__min_watched_pct=0,
            fb__total_dur_min=0
        )
        model.add_user_features(self.user_features)
        model.fit(self.data)
        self.assertEqual(
            {
                ('m',): [1, 2],
                ('f',): [2, 1]
            },
            model.recommendations
        )

    def test_recommend(self):

        model = SegmentRecommender(
            days=1,
            watched_pct_min=0,
            segment=['sex'],
            item_col='iid',
            user_col='uid',
            date_col='ts',
            fb__min_watched_pct=0,
            fb__total_dur_min=0
        )
        model.add_user_features(self.user_features)
        model.fit(self.data)
        df = pd.DataFrame({'uid': self.user_features['uid'].unique().tolist()})
        df['recs'] = model.recommend(df['uid'].tolist(), 2)
        self.assertEqual(
            [
                {'uid': 1, 'recs': [1,2]},
                {'uid': 2, 'recs': [1,2]},
                {'uid': 3, 'recs': [1,2]},
                {'uid': 4, 'recs': [2,1]},
                {'uid': 5, 'recs': [2,1]},
            ],
            df.to_dict('records')
        )
