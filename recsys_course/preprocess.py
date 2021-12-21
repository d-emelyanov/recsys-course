import os
import logging
import pandas as pd
from argparse import ArgumentParser
from common.preprocess import (
    preprocess_interactions,
    preprocess_items,
    preprocess_unused_items,
    preprocess_users
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-j', '--jobs', nargs='*', type=str)
    args = parser.parse_args()
    jobs = args.jobs

    if 'users' in jobs:
        # preprocess users
        users = pd.read_csv(os.path.join(
            BASE_DIR, 'data', 'raw', 'users.csv'
        ))
        users = preprocess_users(users)
        logging.info(
            users.info(verbose=True, null_counts=True)
        )
        users.to_csv(os.path.join(
            BASE_DIR, 'data', 'preprocessed', 'users.csv'
        ), index=None)

    if 'items' in jobs:
        # preprocess items
        items = pd.read_csv(os.path.join(
            BASE_DIR, 'data', 'raw', 'items.csv'
        ))
        items = preprocess_items(items)
        logging.info(
            items.info(verbose=True, null_counts=True)
        )
        items.to_csv(os.path.join(
            BASE_DIR, 'data', 'preprocessed', 'items.csv'
        ), index=None)

    if 'interactions' in jobs:
        # preprocess interactions
        logging.info('Preprocessing interactions')
        interactions = pd.read_csv(os.path.join(
            BASE_DIR, 'data', 'raw', 'interactions.csv'
        ))
        interactions = preprocess_interactions(interactions)
        logging.info(
            interactions.info(verbose=True, null_counts=True)
        )
        interactions.to_csv(os.path.join(
            BASE_DIR, 'data', 'preprocessed', 'interactions.csv'
        ), index=None)

    if 'unused' in jobs:
        logging.info('Preprocessing unused items')
        unused = preprocess_unused_items(interactions, items)
        logging.info(
            interactions.info(verbose=True, null_counts=True)
        )
        unused.to_csv(os.path.join(
            BASE_DIR, 'data', 'preprocessed', 'unused.csv'
        ), index=None)

    if 'item_embeddings' in jobs:
        logging.info('Preprocessing item embeddings')

    if 'user_embeddings' in jobs:
        logging.info('Preprocessing user embeddings')
