from collections import defaultdict

from pycox.evaluation import EvalSurv
from pycox.utils import idx_at_times
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import numpy as np
import pdb
import scipy.stats as st

class Evaluator:
    def __init__(self, df, train_index):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.df_train_all = df.loc[train_index]

    def eval_single(self, model, test_set, val_batch_size=None):
        metric_dict = defaultdict(list)

        df_train_all = self.df_train_all
        get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_train, events_train = get_target(df_train_all)
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                        dtype = [('e', bool), ('t', float)])
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']

        df_test, df_y_test = test_set
        surv = model.interpolate(100).predict_surv_df(df_test)
        durations_test, events_test = get_target(df_y_test)
        et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                    dtype = [('e', bool), ('t', float)])
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        ctd = ev.concordance_td('adj_antolini')
        metric_dict['C-td-full'] = ctd

        # select idx at times
        idx = idx_at_times(surv.index, times, 'post')
        out_survival = surv.iloc[idx].T
        out_risk = 1.0 - out_survival

        # pd -> numpy
        out_risk = np.array(out_risk)

        brs = brier_score(et_train, et_test, out_survival.values, times)[1]

        cis = []
        aucs = []
        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(et_train, et_test, estimate=out_risk[:, i], tau=times[i])[0]
                )
            aucs.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0].item())
            metric_dict[f'{horizons[i]}_Ctd_ipcw'] = cis[i]
            metric_dict[f'{horizons[i]}_brier'] = brs[i]
            metric_dict[f'{horizons[i]}_auroc'] = aucs[i]

        # for horizon in enumerate(horizons):
        #     print(f"For {horizon[1]} quantile,")
        #     print("TD Concordance Index - IPCW:", cis[horizon[0]])
        #     print("Brier Score:", brs[horizon[0]])
        #     print("Dynamic AUC: ", aucs[horizon[0]])
        return metric_dict

    def eval_multi(self, model, test_set, val_batch_size=10000):
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']
        df_train_all = self.df_train_all
        get_target = lambda df, risk: (df['duration'].values, df['event_{}'.format(risk)].values)
        df_test, df_y_test = test_set

        metric_dict = defaultdict(list)
        for risk_idx in range(model.config.num_event):
            durations_train, events_train = get_target(df_train_all, risk_idx)
            durations_test, events_test = get_target(df_y_test, risk_idx)

            surv = model.predict_surv(df_test, batch_size=val_batch_size, event=risk_idx)
            risk = 1 - surv

            et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype = [('e', bool), ('t', float)])
            et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                        dtype = [('e', bool), ('t', float)])

            brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:,1:-1], times)[1]
            cis = []
            for i, _ in enumerate(times):
                cis.append(concordance_index_ipcw(et_train, et_test, risk[:, i+1].to("cpu").numpy(), times[i])[0])
                metric_dict[f'{horizons[i]}_ipcw_{risk_idx}'] = cis[i]
                metric_dict[f'{horizons[i]}_brier_{risk_idx}'] = brs[i]

            for horizon in enumerate(horizons):
                print("Event: {} For {} quantile,".format(risk_idx,horizon[1]))
                print("TD Concordance Index - IPCW:", cis[horizon[0]])
                print("Brier Score:", brs[horizon[0]])

        return metric_dict

    def eval(self, model, test_set, confidence=None, val_batch_size=None, nb_bootstrap=10):
        '''do evaluation.
        if confidence is not None, it should be in (0, 1) and the confidence
        interval will be given by bootstrapping.
        '''
        print("***"*10)
        print("start evaluation")
        print("***"*10)

        if confidence is None:
            if model.config['num_event'] > 1:
                return self.eval_multi(model, test_set, val_batch_size)
            else:
                return self.eval_single(model, test_set, val_batch_size)

        else:
            # do bootstrapping
            stats_dict = defaultdict(list)
            for i in range(nb_bootstrap):
                df_test = test_set[0].sample(test_set[0].shape[0], replace=True)
                df_y_test = test_set[1].loc[df_test.index]

                if model.config['num_event'] > 1:
                    res_dict = self.eval_multi(model, (df_test, df_y_test), val_batch_size)
                else:
                    res_dict = self.eval_single(model, (df_test, df_y_test), val_batch_size)

                for k in res_dict.keys():
                    stats_dict[k].append(res_dict[k])

            metric_dict = {}
            # compute confidence interveal 95%
            alpha = confidence
            p1 = ((1 - alpha) / 2) * 100
            p2 = (alpha + ((1.0 - alpha) / 2.0)) * 100
            for k in stats_dict.keys():
                stats = stats_dict[k]
                lower = max(0, np.percentile(stats, p1))
                upper = min(1.0, np.percentile(stats, p2))
                metric_dict[k] = [(upper + lower) / 2, (upper - lower) / 2]

            return metric_dict
