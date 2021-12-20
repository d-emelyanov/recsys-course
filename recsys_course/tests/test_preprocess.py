import pandas as pd
from  unittest import TestCase

from pandas._testing.asserters import assert_is_sorted
from preprocess import preprocess_users, onehot
from pandas.testing import assert_frame_equal


class TestPreprocess(TestCase):

    def test_onehot(self):
        a = pd.DataFrame({'sex': ['m', 'zh', 'm']})
        assert_frame_equal(
            pd.DataFrame({
                'sex': ['m', 'zh', 'm'],
                'sex_m': [1, 0, 1],
                'sex_zh': [0, 1, 0]
            }),
            onehot(a, ['sex'], True),
            check_column_type=False,
            check_dtype=False
        )

    def test_users(self):
        data = pd.DataFrame([
            (1, 'age_25_34', 'income_60_90', 'М', 1),
            (2, 'age_18_24','income_20_40', 'М', 0),
            (3, 'age_45_54', 'income_40_60', 'Ж', 1),
            (4, 'age_35_44', 'income_0_20', 'Ж', 0),
            (5, pd.NA, pd.NA, pd.NA, pd.NA,),
            (6, 'age_55_64', 'income_150_inf', 'М', 1),
            (7, 'age_65_inf',  'income_150_inf', 'Ж', 0)
        ], columns=['user_id', 'age', 'income', 'sex', 'kids_flg'])

        test = pd.DataFrame([
            (
                1, 'age_25_34', 'income_60_90', 'm', 1,
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0
            ),
            (
                2, 'age_18_24','income_20_40', 'm', 0,
                0, 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0,
                1, 0, 0
            ),
            (
                3, 'age_45_54', 'income_40_60', 'zh', 1,
                0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0,
                0, 1, 0
            ),
            (
                4, 'age_35_44', 'income_0_20', 'zh', 0,
                0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0,
                0, 1, 0
            ),
            (
                5, 'age_unknown', 'income_unknown', 'unknown', -1,
                0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 1
            ),
            (
                6, 'age_55_64', 'income_150_inf', 'm', 1,
                0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1,
                1, 0, 0
            ),
            (
                7, 'age_65_inf',  'income_150_inf', 'zh', 0,
                0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 1,
                0, 1, 0
            )
        ], columns=[
            'user_id', 'age', 'income', 'sex', 'kids_flg',
            'age_25_34', 'age_18_24', 'age_45_54', 'age_35_44', 'age_unknown', 'age_55_64', 'age_65_inf',
            'income_60_90', 'income_20_40', 'income_40_60', 'income_0_20', 'income_unknown', 'income_150_inf',
            'sex_m', 'sex_zh', 'sex_unknown'
        ])

        assert_frame_equal(
            preprocess_users(data),
            test,
            check_column_type=False,
            check_dtype=False,
            check_like=True
        )