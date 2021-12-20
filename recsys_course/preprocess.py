import argparse
import os
import logging
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from transliterate import translit
from common.doc2vec import Doc2Vec


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def onehot(df, cols, prefix=False):
    for c in cols:
        if prefix:
            prefix_ = {
                'prefix': c
            }
        else:
            prefix_ = {}
        df = pd.concat([
            df,
            pd.get_dummies(df[c], **prefix_)
        ], axis=1)
    return df


def preprocess_items(data):
    logging.info('Preprocessing items')
    data['content_type'] = data['content_type'].astype('category')
    data['title'] = data['title'].str.lower()
    data['title_orig'] = data['title_orig'].fillna('None')

    data.loc[data['release_year'].isna(), 'release_year'] = 2020.
    data.loc[data['release_year'] < 1920, 'release_year_cat'] = 'inf_1920'
    data.loc[data['release_year'] >= 2020, 'release_year_cat'] = '2020_inf'
    for i in range (1920, 2020, 10):
        data.loc[
            (data['release_year'] >= i) & (data['release_year'] < i+10),
            'release_year_cat'
        ] = f'{i}-{i+10}'
    data['release_year_cat'] = data['release_year_cat'].astype('category')

    data['genres'] = data['genres'].astype('category')
    data.loc[data.countries.isna(), 'countries'] = 'Россия'
    data['countries'] = data['countries'].str.lower()
    data['countries'] = data['countries'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
    data['countries'] = data['countries'].astype('category')

    # TODO: there is a smarter way to do it
    data['for_kids'] = data['for_kids'].fillna(-1)
    data['for_kids'] = data['for_kids'].astype('int')

    data['for_kids_rating'] = data['age_rating'].map(
        lambda x: 1 if x <= 6 else -1 if pd.isna(x) else 0
    )
    data.loc[data.age_rating.isna(), 'age_rating'] = 0
    data['age_rating'] = data['age_rating'].astype('category')

    data['studios'] = data['studios'].fillna('Unknown')
    data['studios'] = data['studios'].str.lower()
    data['studios'] = data['studios'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
    data['studios'] = data['studios'].astype('category')

    data['directors'] = data['directors'].fillna('Unknown')
    data['directors'] = data['directors'].str.lower()
    data['directors'] = data['directors'].astype('category')

    data['actors'] = data['actors'].fillna('Unknown')
    data['actors'] = data['actors'].astype('category')

    data['keywords'] = data['keywords'].fillna('Unknown')
    data['keywords'] = data['keywords'].astype('category')

    data['description'] = data['description'].fillna('-')

    data = onehot(
        data,
        cols=['content_type'],
        prefix=True
    )

    return data


def preprocess_unused_items(data, items):

    class UnusedItems:
        def __init__(self, ids: set, r: int):
            self.ids = ids
            self.r = r

        def __call__(self, x):
            l = list(set(list(x)) ^ self.ids)
            return np.random.choice(l, min([len(l), self.r]))

    items = set(data['item_id'].unique().tolist()).intersection(
        set(items['item_id'].unique().tolist())
    )
    unused = (
        data
        .groupby('user_id')['item_id']
        .apply(UnusedItems(
            ids=items,
            r=30
        ))
        .reset_index()
        .explode('item_id')
        .dropna()
    )
    return unused

def preprocess_users(data):
    logging.info('Preprocessing users')

    data['age'] = data['age'].fillna('age_unknown').astype(str)
    data['income'] = data['income'].fillna('income_unknown').astype(str)
    data['sex'] = data['sex'].fillna('unknown').astype(str)
    data['kids_flg'] = data['kids_flg'].fillna(-1).astype('int8')

    data['sex'] = data['sex'].map(lambda x: translit(x.lower(), 'ru', reversed=True))

    data = onehot(data, ['age', 'income'])
    data = onehot(data, ['sex'], prefix=True)

    return data


def preprocess_interactions(data):

    data['watched_pct'] = data['watched_pct'].astype(pd.Int8Dtype())
    data['watched_pct'] = data['watched_pct'].fillna(0)
    data['last_watch_dt'] = pd.to_datetime(data['last_watch_dt'])

    return data


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('-j', '--job', nargs='*', type=str)
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
