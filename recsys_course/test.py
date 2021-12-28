import importlib
import logging
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from common.data import DataLoader
from models.loader import load_model
from common.metrics import map_at_k
from .const import *


if __name__ == '__main__':


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-fb', '--fallback', type=str, default=None)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--n_recs', type=int, default=10)
    parser.add_argument('--watched_pct_min', type=int, default=0)
    parser.add_argument('-t', '--test_file', type=str, required=True)
    parser.add_argument('-s', '--submission', type=str, required=True)
    args, params = parser.parse_known_args()

    res_df = []

    logging.info('Loading data..')
    data = DataLoader.from_folder(
        args.data,
        watched_pct_min=args.watched_pct_min,
        user_col=USER_COL,
        item_col=ITEM_COL,
        date_col=DATE_COL
    )

    test = pd.read_csv(args.test_file)

    # Warm Users
    test_warm = test.loc[
        test[USER_COL].isin(
            data.interactions[USER_COL].unique().tolist()
        )
    ]

    rec = load_model(
        path=args.recsys,
        params=params,
        user_col=USER_COL,
        item_col=ITEM_COL,
        date_col=DATE_COL
    )
    rec.add_item_features(data.items)
    rec.add_user_features(data.users)
    rec.fit(
        data.interactions.loc[
            data.interactions.watched_pct >= args.watched_pct_min
        ]
    )

    logging.info('Generating warm predictions')
    df = pd.DataFrame({USER_COL: test_warm[USER_COL].unique().tolist()})
    df['item_id'] = rec.recommend(
        test_warm[USER_COL].tolist(),
        N=args.n_recs
    )
    res_df.append(df)

    # Cold Users
    test_cold = test.loc[
        ~test[USER_COL].isin(
            data.interactions[USER_COL].unique().tolist()
        )
    ]
    logging.info('Generating cold predictions')
    df = pd.DataFrame({USER_COL: test_cold[USER_COL].unique().tolist()})
    if args.fallback is not None:
        fb = load_model(
            path=args.fallback,
            params=params,
            user_col=USER_COL,
            item_col=ITEM_COL,
            date_col=DATE_COL
        )
        fb.add_item_features(data.items)
        fb.add_user_features(data.users)
        fb.fit(data.interactions)
        df['item_id'] = fb.recommend(
            test_cold[USER_COL].tolist(),
            N=args.n_recs
        )
    else:
        df['item_id'] = rec.recommend(
            test_warm[USER_COL].tolist(),
            N=args.n_recs
        )
    res_df.append(df)

    res_df = pd.concat(res_df)

    if res_df.shape[0] != test.shape[0]:
        raise ValueError('Size mismatch')

    res_df.to_csv(args.submission, index=False)
