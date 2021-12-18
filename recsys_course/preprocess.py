import os
import logging
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def preprocess_items():
    logging.info('Preprocessing items')
    data = pd.read_csv(os.path.join(
        BASE_DIR, 'data', 'raw', 'items.csv'
    ))

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
    data['for_kids'] = data['for_kids'].fillna(0)
    data['for_kids'] = data['for_kids'].astype('bool')

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

    logging.info(
        data.info(verbose=True, null_counts=True)
    )

    data.to_csv(os.path.join(
        BASE_DIR, 'data', 'preprocessed', 'items.csv'
    ), index=None)


def preprocess_users():
    logging.info('Preprocessing users')
    data = pd.read_csv(os.path.join(
        BASE_DIR, 'data', 'raw', 'users.csv'
    ))

    data['age'] = data['age'].fillna('age_unknown').astype('category')
    data['income'] = data['income'].fillna('income_unknown').astype('category')
    data['sex'] = data['sex'].fillna('sex_unknown').astype('category')
    data['kids_flg'] = data['kids_flg'].astype('bool')

    logging.info(
        data.info(verbose=True, null_counts=True)
    )

    data.to_csv(os.path.join(
        BASE_DIR, 'data', 'preprocessed', 'users.csv'
    ), index=None)


def preprocess_interactions():
    logging.info('Preprocessing interactions')
    data = pd.read_csv(os.path.join(
        BASE_DIR, 'data', 'raw', 'interactions.csv'
    ))

    data['watched_pct'] = data['watched_pct'].astype(pd.Int8Dtype())
    data['watched_pct'] = data['watched_pct'].fillna(0)

    data['last_watch_dt'] = pd.to_datetime(data['last_watch_dt'])

    logging.info(
        data.info(verbose=True, null_counts=True)
    )

    data.to_csv(os.path.join(
        BASE_DIR, 'data', 'preprocessed', 'interactions.csv'
    ), index=None)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    preprocess_users()
    preprocess_items()
    preprocess_interactions()
