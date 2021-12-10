import argparse
import importlib
import mlflow
from argparse import ArgumentParser
from  common.data import DataLoader


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    args, params = parser.parse_known_args()

    data = DataLoader.from_folder(args.data)

    rec_module = '.'.join(args.recsys.split('.')[:-1])
    module = importlib.import_module(f'models.{rec_module}')

    rec_class_name = args.recsys.split('.')[-1]
    rec_class = getattr(module, rec_class_name)
    rec = rec_class.from_args(params)

    rec.fit(data)

    mlflow.set_experiment('recsys')
    with mlflow.start_run():

        mlflow.log_param('x', 1)
        mlflow.log_metric('y', 2)
