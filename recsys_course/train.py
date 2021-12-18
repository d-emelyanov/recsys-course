import importlib
import mlflow
import logging
import pandas as pd
import numpy as np
import time
from argparse import ArgumentParser
from datetime import datetime
from common.data import DataLoader
from common.tuning import Optimizer
from common.metrics import map_at_k
from .const import *


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--n_recs', type=int, required=True)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    parser.add_argument('--optuna_trials', type=int, default=1)
    args, params = parser.parse_known_args()

    logging.info('Loading data..')
    data = DataLoader.from_folder(
        args.data,
        user_col=USER_COL,
        item_col=ITEM_COL,
        date_col=DATE_COl
    )

    rec_module = '.'.join(args.recsys.split('.')[:-1])
    module = importlib.import_module(f'models.{rec_module}')
    rec_class_name = args.recsys.split('.')[-1]
    rec_class = getattr(module, rec_class_name)

    mlflow.set_experiment('recsys')
    start_time = time.time()
    with mlflow.start_run():

        if args.optuna:
            logging.info('Start using Optuna..')
            optimizer = Optimizer.from_args(params)
            optimizer.optimize(
                model=rec_class,
                data=data,
                n_recs=args.n_recs,
                test_size=args.test_size,
                trials=args.optuna_trials
            )
            metrics = optimizer.best_metrics
            log_params = optimizer.best_params
        else:
            logging.info('Training model..')

            rec = rec_class.from_args(
                params,
                user_col=USER_COL,
                item_col=ITEM_COL,
                date_col=DATE_COl
            )
            rec.add_item_features(data.items)
            rec.add_user_features(data.users)

            train, test = data.get_train_test(args.test_size)
            rec.fit(train)

            df = pd.DataFrame({USER_COL: test[USER_COL].unique().tolist()})
            df = pd.merge(
                left=df,
                right=data.get_real(test).rename(columns={ITEM_COL: 'real'}),
                on=[USER_COL]
            )

            df['recs'] = rec.recommend(
                df[USER_COL].tolist(),
                N=args.n_recs
            )

            metrics = {
                f'map{args.n_recs}': map_at_k(
                    k=args.n_recs,
                    recs=df['recs'],
                    real=df['real']
                )
            }
            log_params = rec.params

        end_time = time.time()

        mlflow.log_metrics({
            **metrics,
            'time_exec': end_time - start_time
        })

        mlflow.log_params({
            **log_params,
            **{
                'test_size': args.test_size,
                'data': args.data,
                'recsys': args.recsys,
                'optuna': args.optuna,
                'optuna_trials': args.optuna_trials,
                'params': str(params),
                'n_recs': args.n_recs
            }
        })
