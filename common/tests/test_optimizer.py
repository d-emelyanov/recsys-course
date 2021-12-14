from common.tuning import Optimizer
from unittest import TestCase
from argparse import Namespace


class TestOptimizer(TestCase):

    def test_from_args(self):
        args = [
            '--x__type', 'float',
            '--x__low', '10',
            '--x__high', '15',
            '--y__type', 'int',
            '--y__low', '1',
            '--y__high', '5',
            '--z__type', 'float',
            '--z__fixed', '10'
        ]
        opt = Optimizer.from_args(args)
        self.assertDictEqual(
            opt.trial_params,
            {
                'x': ('suggest_float', {'low': 10, 'high': 15}),
                'y': ('suggest_int', {'low': 1, 'high': 5})
            }
        )
        self.assertDictEqual(
            opt.fixed_params,
            {'z': 10}
        )

