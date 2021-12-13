import argparse
import importlib
import mlflow
from argparse import ArgumentParser
from datetime import datetime
from common.data import DataLoader
from common.tuning import Optimizer
from common.metrics import map_at_k


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--n_recs', type=int, required=True)
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    args, params = parser.parse_known_args()

    data = DataLoader.from_folder(args.data)
    data.train_test_split()

    rec_module = '.'.join(args.recsys.split('.')[:-1])
    module = importlib.import_module(f'models.{rec_module}')
    rec_class_name = args.recsys.split('.')[-1]
    rec_class = getattr(module, rec_class_name)

    mlflow.set_experiment('recsys')
    with mlflow.start_run():

        if args.optuna:
            optimizer = Optimizer.from_args(params)
            optimizer.optimize(
                model=rec_class,
                data=data,
                n_recs=args.n_recs
            )
            rec = optimizer.best_model
        else:
            rec = rec_class.from_args(params)
            rec.fit(data.train)

        mlflow.log_metric(
            f'train_map@{args.n_recs}',
            map_at_k(
                k=args.n_recs,
                recs=rec.predict(data.train),
                real=data.train['real']
            )
        )

        mlflow.log_metrics(
            f'test_map@{args.n_recs}',
            map_at_k(
                k=args.n_recs,
                recs=rec.predict(data.test),
                real=data.test['real']
            )
        )
