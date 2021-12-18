import importlib
import logging
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from common.data import DataLoader
from common.metrics import map_at_k
from .const import *


if __name__ == '__main__':


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--n_recs', type=int, required=True)
    parser.add_argument('-t', '--test_file', type=str, required=True)
    parser.add_argument('-s', '--submission', type=str, required=True)
    args, params = parser.parse_known_args()

    logging.info('Loading data..')
    data = DataLoader.from_folder(
        args.data,
        user_col=USER_COL,
        item_col=ITEM_COL,
        date_col=DATE_COl
    )

    test = pd.read_csv(args.test_file)

    rec_module = '.'.join(args.recsys.split('.')[:-1])
    module = importlib.import_module(f'models.{rec_module}')
    rec_class_name = args.recsys.split('.')[-1]
    rec_class = getattr(module, rec_class_name)

    logging.info('Training model..')
    rec = rec_class.from_args(
        params,
        user_col=USER_COL,
        item_col=ITEM_COL,
        date_col=DATE_COl
    )
    rec.add_item_features(data.items)
    rec.add_user_features(data.users)
    rec.fit(data.interactions)

    logging.info('Generating predictions')
    df = pd.DataFrame({USER_COL: test[USER_COL].unique().tolist()})
    df['item_id'] = rec.recommend(
        test[USER_COL].tolist(),
        N=args.n_recs
    )
    df.to_csv(args.submission, index=False)
