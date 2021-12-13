import importlib
import mlflow
import logging
from argparse import ArgumentParser
from datetime import datetime
from common.data import DataLoader
from common.tuning import Optimizer
from common.metrics import map_at_k


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--n_recs', type=int, required=True)
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    args, params = parser.parse_known_args()

    logging.info('Loading data..')
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
            logging.info('Training model..')
            rec = rec_class.from_args(params)
            rec.fit(data.train)


        mlflow.log_metric(
            f'train_map{args.n_recs}',
            map_at_k(
                k=args.n_recs,
                recs=rec.recommend(data.train, N=args.n_recs)['recs'],
                real=data.train_real
            )
        )

        mlflow.log_metric(
            f'test_map{args.n_recs}',
            map_at_k(
                k=args.n_recs,
                recs=rec.recommend(data.test, N=args.n_recs)['recs'],
                real=data.test_real
            )
        )
