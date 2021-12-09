from common.abstract import BaseRecommender
from argparse import ArgumentParser


class PopularRecommender(BaseRecommender):

    @classmethod
    def from_args(cls, args):
        parser = ArgumentParser()
        parser.add_argument('-k', type=int)
        parser.add_argument('-days', type=int)
        args = parser.parse_args(args)
        return cls(
            k=args.k,
            days=args.days
        )

    def __init__(
        self,
        k,
        days
    ):
        self.k = k
        self.days = days

    def fit(self):
        pass

    def recommend(self):
        pass

    def test(self):
        print('hui')
        print(self.k, self.days)