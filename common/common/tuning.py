import optuna
import pandas as pd
import numpy as np
from .metrics import map_at_k


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
        rec = self.model(
            **self.fixed_params,
            **self.best_params
        )
        rec.fit(self.data.train)
        return rec

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

    def train(self, rec):
        steps = []
        for train, test, _ in self.data.get_folds(
            self.folds, self.fold_size
        ):
            rec.fit(train)
            df = pd.DataFrame({
                self.data.user_col: (
                    test[self.data.user_col]
                    .unique()
                    .tolist()
                )
            })
            df = pd.merge(
                left=df,
                right=(
                    self.data
                    .get_real(test)
                    .rename(columns={self.data.item_col: 'real'})
                ),
                on=[self.data.user_col]
            )
            df['recs'] = rec.recommend(
                df[self.data.user_col], N=self.n_recs
            )
            steps.append({
                f'map{self.n_recs}': map_at_k(
                    k=self.n_recs,
                    recs=df['recs'],
                    real=df['real']
                )
            })
        return steps

    @property
    def best_params(self):
        return self.get_params(self.study.best_trial)

    @property
    def best_metrics(self):
        return self.detailed_objective(
            self.study.best_trial
        )

    def detailed_objective(self, trial):
        rec = self.model(
            **self.get_params(trial),
            user_col=self.data.user_col,
            item_col=self.data.item_col,
            date_col=self.data.date_col
        )
        steps = self.train(rec)
        return {
            f'map{self.n_recs}': np.mean([
                x[f'map{self.n_recs}'] for x in steps
            ])
        }

    def objective(self, trial):
        rec = self.model(
            **self.get_params(trial),
            user_col=self.data.user_col,
            item_col=self.data.item_col,
            date_col=self.data.date_col
        )
        steps = self.train(rec)
        return np.mean([x[f'map{self.n_recs}'] for x in steps])

    def optimize(self, n_recs, model, data, folds, fold_size, trials):
        self.study = optuna.create_study(direction='maximize')

        self.n_recs = n_recs
        self.model = model
        self.data = data
        self.folds = folds
        self.fold_size = fold_size

        self.study.optimize(
            self.objective,
            n_trials=trials
        )
