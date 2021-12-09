from argparse import ArgumentParser
import argparse
import importlib
from  common.data import DataLoader


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-r', '--recsys', type=str, required=True)
    parser.add_argument('-d', '--data', nargs='*', required=True)
    parser.add_argument('-p', '--params', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    data = DataLoader.from_list(args.data)

    rec_module = '.'.join(args.recsys.split('.')[:-1])
    module = importlib.import_module(f'models.{rec_module}')

    rec_class_name = args.recsys.split('.')[-1]
    rec_class = getattr(module, rec_class_name)
    rec = rec_class.from_args(args.params)

    rec.fit(data)
