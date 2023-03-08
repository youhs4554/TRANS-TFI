import numpy as np
from pycox import utils
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from btdsa.train_utils import get_target
from btdsa.utils import bootstrap_eval
import prettytable as pt


class EvalSurv:
    def __init__(self, trainer=None):
        self.trainer = trainer
        self.headers = []
        self.results = []

    def evaluate(self, surv, x_test, y_test):
        durations_test, events_test = y_test.T

        assert hasattr(self.trainer, "trained_model"), "You need to fit() model first. The model is not fitted yet!"

        # bootstrap evaluation
        result = bootstrap_eval(self.trainer.trained_model, x_test, durations_test, events_test,
                                interpolate_discrete_times=self.trainer.interpolate_discrete_times,
                                nb_bootstrap=self.trainer.cfg.nb_bootstrap)

        # for beautified result table
        self.headers.append(self.trainer.dataset)
        row_str = f"{result['mean']:.6f} ({result['confidence_interval'][0]:.6f},{result['confidence_interval'][1]:.6f})\n"

        # eval IPCW/Brier/AUC
        t_train, e_train = get_target(self.trainer.df_train_raw)
        t_val, e_val = get_target(self.trainer.df_val_raw)
        t_test, e_test = get_target(self.trainer.df_test_raw)

        cis = []
        brs = []

        et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                            dtype=[('e', bool), ('t', float)])
        et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                           dtype=[('e', bool), ('t', float)])
        et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                          dtype=[('e', bool), ('t', float)])

        taus = self.trainer.taus
        idx = utils.idx_at_times(surv.index, taus, 'post')
        out_survival = surv.iloc[idx].T
        out_risk = 1.0 - out_survival

        # pd -> numpy
        out_survival = np.array(out_survival)
        out_risk = np.array(out_risk)

        for i, _ in enumerate(taus):
            cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], taus[i])[0])
        brs.append(brier_score(et_train, et_test, out_survival, taus)[1])
        roc_auc = []
        for i, _ in enumerate(taus):
            roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], taus[i])[0].item())

        for i, horizon in enumerate(self.trainer.cfg.horizons):
            row_str += f"[{horizon * 100}%] C-td: {cis[i]:.6f}, Brier: {brs[0][i]:.6f}, AUC-td: {roc_auc[i]:.6f}\n"
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
