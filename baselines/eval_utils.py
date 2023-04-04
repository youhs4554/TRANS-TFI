from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
import prettytable as pt
import torch
from pycox.evaluation import EvalSurv
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc

from baselines.train_utils import get_target
from pycox.utils import idx_at_times


@torch.no_grad()
def bootstrap_eval(model, x_test, durations_test, events_test, et_train, taus, horizons,
                   interpolate_discrete_times=True, nb_bootstrap=100):
    nb_samples = len(x_test)
    result_dict = defaultdict(list)  # C-td(IPCW)/Brier/AUC of 25%,50%,75% horizons

    def resample():
        x_test_res = pd.DataFrame(x_test).sample(nb_samples, replace=True)
        durations_test_res = pd.Series(durations_test).loc[x_test_res.index]
        events_test_res = pd.Series(events_test).loc[x_test_res.index]
        return x_test_res.values, durations_test_res.values, events_test_res.values

    for i in range(nb_bootstrap):
        x_test_res, durations_test_res, events_test_res = resample()
        if interpolate_discrete_times:
            surv = model.interpolate(100).predict_surv_df(x_test_res)
        else:
            surv = model.predict_surv_df(x_test_res)
        ev = EvalSurv(surv, durations_test_res, events_test_res, censor_surv='km')
        cindex = ev.concordance_td('adj_antolini')
        result_dict['C-td-full'].append(cindex)

        # select idx at times
        idx = idx_at_times(surv.index, taus, 'post')
        out_survival = surv.iloc[idx].T
        out_risk = 1.0 - out_survival

        et_test = np.array([(events_test_res[i], durations_test_res[i]) for i in range(len(events_test_res))],
                           dtype=[('e', bool), ('t', float)])

        metric_dict = {}
        cis = []
        brs = brier_score(et_train, et_test, out_survival.values, taus)[1]
        aucs = []

        # pd -> numpy
        out_risk = np.array(out_risk)

        for i, _ in enumerate(taus):
            cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], taus[i])[0])
            aucs.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], taus[i])[0].item())

            metric_dict[f'{horizons[i]}_Ctd_ipcw'] = cis[i]
            metric_dict[f'{horizons[i]}_brier'] = brs[i]
            metric_dict[f'{horizons[i]}_auroc'] = aucs[i]

        for k in metric_dict.keys():
            result_dict[k].append(metric_dict[k])

    confi_dict = {}
    # compute confidence interveal 95%
    alpha = 0.95
    p1 = ((1 - alpha) / 2) * 100
    p2 = (alpha + ((1.0 - alpha) / 2.0)) * 100
    for k in result_dict.keys():
        stats = result_dict[k]
        lower = max(0, np.percentile(stats, p1))
        upper = min(1.0, np.percentile(stats, p2))
        confi_dict[k] = [(upper + lower) / 2, (upper - lower) / 2]
    return confi_dict


class Evaluator:
    def __init__(self, trainer=None):
        self.trainer = trainer
        self.headers = []
        self.results = []

    def evaluate(self, x_test, y_test):
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

        cindex_avg, cindex_interval = result_dict.pop('C-td-full')
        row_str = f"C-td (full): {cindex_avg:.4f} ({cindex_interval:.4f})\n"
        mlflow.log_metric('Ctd__avg', cindex_avg)
        mlflow.log_metric('Ctd__interval', cindex_interval)

        for horizon in horizons:
            keys = [k for k in result_dict.keys() if k.startswith(str(horizon))]
            results_at_horizon = [result_dict[k] for k in keys]
            msg = [f"[{round(horizon * 100, 4)}%]"]
            for k, res in zip(keys, results_at_horizon):
                metric = k[k.find('_')+1:]
                avg, interval = res
                msg.append(f"{metric}: {avg:.4f} ({interval:.4f})")
                mlflow.log_metric(str(horizon) + '_' + metric + '__avg', avg)
                mlflow.log_metric(str(horizon) + '_' + metric + '__interval', interval)
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
