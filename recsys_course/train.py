import importlib
import mlflow
import logging
import pandas as pd
import numpy as np
import time
from argparse import ArgumentError, ArgumentParser
from datetime import date, datetime
from common.data import DataLoader
from common.tuning import Optimizer
from .trainer import Trainer, get_model_class
from .const import *


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-fb', '--fallback', type=str, default=None)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--n_recs', type=int,  default=10)
    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--folds', type=int, default=None)
    parser.add_argument('--watched_pct_min', type=int, default=0)
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    parser.add_argument('--optuna_trials', type=int, default=1)
    args, params = parser.parse_known_args()

    logging.info('Loading data..')

    data = DataLoader.from_folder(
        args.data,
        watched_pct_min=args.watched_pct_min,
        user_col=USER_COL,
        item_col=ITEM_COL,
        date_col=DATE_COL
    )

    rec_class = get_model_class(args.recsys)
    if args.fallback:
        fb_class = get_model_class(args.fallback)
    else:
        fb_class = None
    mlflow.set_experiment('recsys')
    start_time = time.time()

    trainer = Trainer(
        params=params,
        rec_class=rec_class,
        fb_class=fb_class,
        data=data,
        n_recs=args.n_recs,
        test_size=args.test_size,
        folds=args.folds
    )

    with mlflow.start_run():

        if args.optuna:
            logging.info('Start using Optuna..')
            optimizer = Optimizer.from_args(params)
            optimizer.optimize(
                trials=args.optuna_trials,
                train_func=trainer.train,
            )
            metrics = optimizer.best_metrics
            log_params = optimizer.best_params
        else:
            logging.info('Training model..')
            metrics, log_params, rec = trainer.train()

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
                'watched_pct_min': args.watched_pct_min,
                'n_recs': args.n_recs
            }
        })
