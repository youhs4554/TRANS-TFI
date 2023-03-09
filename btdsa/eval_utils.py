import numpy as np
import prettytable as pt

from btdsa.train_utils import get_target
from btdsa.utils import bootstrap_eval


class EvalSurv:
    def __init__(self, trainer=None):
        self.trainer = trainer
        self.headers = []
        self.results = []

    def evaluate(self, surv, x_test, y_test):
        durations_test, events_test = y_test.T

        assert hasattr(self.trainer, "trained_model"), "You need to fit() model first. The model is not fitted yet!"

        t_train, e_train = get_target(self.trainer.df_train_raw)
        et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                            dtype=[('e', bool), ('t', float)])
        taus = self.trainer.taus
        horizons = self.trainer.cfg.horizons

        # bootstrap evaluation
        result_dict = bootstrap_eval(self.trainer.trained_model, x_test, durations_test, events_test, et_train,
                                     taus, horizons,
                                     interpolate_discrete_times=self.trainer.interpolate_discrete_times,
                                     nb_bootstrap=self.trainer.cfg.nb_bootstrap)

        # for beautified result table
        self.headers.append(self.trainer.dataset)

        cindex_mean, (cindex_lower, cindex_upper) = result_dict.pop('C-td-full')

        row_str = f"C-td (full): {cindex_mean:.6f} ({cindex_lower:.6f},{cindex_upper:.6f})\n"

        for horizon in horizons:
            keys = [ k for k in result_dict.keys() if k.startswith(str(horizon)) ]
            results_at_horizon = [result_dict[k] for k in keys]
            msg = [f"[{horizon*100}%]"]
            for k,res in zip(keys,results_at_horizon):
                metric = k.split('_')[1]
                mean, (lower, upper) = res
                msg.append(f"{metric}: {mean:.6f} ({lower:.6f},{upper:.6f})")
            row_str += (" ".join(msg) + "\n")
        self.results.append(row_str)

    def report(self):
        cfg = self.trainer.cfg
        if self.trainer.interpolate_discrete_times:
            title = f"{cfg.model_name}({cfg.time_range}, L={cfg.seq_len})"
        else:
            title = cfg.model_name

        tb = pt.PrettyTable(title=title)
        tb.field_names = self.headers
        tb.add_row(self.results)
        self.trainer.logger.info(tb)
