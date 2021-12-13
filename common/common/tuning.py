import optuna
from .metrics import map_at_k


class Optimizer:

    @classmethod
    def from_args(cls, params):
        vars_ = {}
        fixed = {}
        params = vars(params)
        for pn, pv in params.items():
            if '__' not in pn:
                fixed[pn] = pv
            else:
                var_, var_t = pn.split('__')
                if var_ not in vars_.keys():
                    vars_[var_] = {
                        'method': None,
                        'params': {}
                    }
                if var_t == 'type':
                    vars_[var_]['method'] = f'suggest_{pv}'
                else:
                    vars_[var_]['params'][var_t] = pv
        return cls(
            fixed_params=fixed,
            trial_params={
                k: (v['method'], v['params'])
                for k,v in vars_.items()
            }
        )

    def __init__(self, fixed_params, trial_params):
        self.fixed_params = fixed_params
        self.trial_params = trial_params

    def get_params(self, trial):
        params = {}
        for var, (method, params) in self.params.items():
            params[var] = getattr(trial, method)(**params)
        return {
            **params,
            **self.fixed_params
        }

    def objective(self, trial):
        rec = self.model(**self.get_params(trial))
        rec.fit(self.data.train)
        return map_at_k(
            k=self.n_recs,
            recs=rec.predict(self.data.train),
            real=self.data.train['real']
        )

    def optimize(self, n_recs, model, data):
        study = optuna.create_study()

        self.n_recs = n_recs
        self.model = model
        self.data = data

        study.optimize(
            self.objective,
            n_trials=100
        )
