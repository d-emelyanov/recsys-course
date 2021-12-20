import pandas as pd
import numpy as np

from gensim.models.doc2vec import (
    Doc2Vec as GensimDoc2Vec,
    TaggedDocument
)


def make_doc(x):
    '''
    Function makes a document from a list of actions. Can be used
    in pandas group by
    '''
    actions = x.sort_values('event_timestamp')['action'].tolist()
    return pd.Series({
        'doc_len': len(actions),
        'doc': actions
    })


class MakeDoc:

    def __init__(self, time_col, action_col):
        self.time_col = time_col
        self.action_col = action_col

    def __call__(self, x):
        actions = x.sort_values(self.time_col)[self.action_col].tolist()
        return pd.Series({
            'doc_len': len(actions),
            'doc': actions
        })


class Doc2Vec:

    def __init__(
        self,
        vector_size=50,
        min_count=2,
        epochs=50,
        doc_column='doc'
    ):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.doc_column = doc_column

    def fit(self, data):

        corp = []
        for i, r in enumerate(data.to_dict('records')):
            corp.append(TaggedDocument(r[self.doc_column], [i]))

        self.model = GensimDoc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.min_count
        )
        self.model.build_vocab(corp)
        self.model.train(
            corp,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )

    def transform(self, data):
        arr = []
        for doc in data[self.doc_column].tolist():
            arr.append(self.model.infer_vector(doc))
        return np.array(arr)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
