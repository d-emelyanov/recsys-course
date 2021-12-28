import optuna
import pandas as pd
import numpy as np
from .metrics import map_at_k
from functools import partial


class Optimizer:

    @classmethod
    def from_args(cls, params_string):
        params = {}
        funcs = {
            'float': float,
            'int': int,
        }
        for i in range(0, len(params_string), 2):
            k, kv = params_string[i].replace('--', '').split('__')
            v = params_string[i+1]
            if k not in params:
                params[k] = {}
            if kv not in params[k]:
                params[k][kv] = v

        for k in params:
            for p in params[k]:
                if p != 'type':
                    params[k][p] = funcs.get(params[k]['type'], str)(params[k][p])

        vars_ = {}
        fixed = {}
        for pn, pv in params.items():
            if 'fixed' in pv:
                fixed[pn] = pv['fixed']
            else:
                vars_[pn] = {
                    'method': f"suggest_{pv['type']}",
                    'params': {
                        pvk: pvv
                        for pvk, pvv in pv.items()
                        if pvk != 'type'
                    }
                }

        return cls(
            fixed_params=fixed,
            trial_params={
                k: (v['method'], v['params'])
                for k, v in vars_.items()
            }
        )

    @property
    def best_model(self):
        _, _, rec = self.train(
            params={
                **self.fixed_params,
                **self.best_params
            }
        )
        return rec

    @property
    def best_params(self):
        return self.get_params(self.study.best_trial)

    @property
    def best_metrics(self):
        return self.detailed_objective(
            self.study.best_trial
        )

    def __init__(self, fixed_params, trial_params):
        self.fixed_params = fixed_params
        self.trial_params = trial_params

    def get_params(self, trial):
        params = {}
        for var, (method, params_) in self.trial_params.items():
            params[var] = getattr(trial, method)(var, **params_)
        return {
            **params,
            **self.fixed_params
        }

    def detailed_objective(self, trial):
        model_params = self.get_params(trial)
        self.trainer.update_params(model_params)
        metrics, _, (_, _) = self.trainer.train()
        return metrics

    def objective(self, trial):
        model_params = self.get_params(trial)
        self.trainer.update_params(model_params)
        metrics, _, (_, _) = self.trainer.train()
        return metrics['map10']

    def optimize(self, trials, trainer):
        self.study = optuna.create_study(direction='maximize')
        self.trainer = trainer

        self.study.optimize(
            self.objective,
            n_trials=trials
        )
