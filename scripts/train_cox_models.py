from collections import defaultdict

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc
import prettytable as pt

from baselines.datasets import load_data
from baselines.utils import seed_everything

DATASETS = ['gbsg', 'metabric', 'flchain']

MODEL_DICT = {
    "cph": CoxPHSurvivalAnalysis,
    "rsf": RandomSurvivalForest
}

horizons = [.25, .5, .75]
random_state = 1234
nb_bootstrap = 10


def format_labels(y):
    durations, events = y.T
    return np.array([(events[i], durations[i]) for i in range(len(y))],
                    dtype=[('e', bool), ('t', float)])


def evaluate(model, x_test, y_test, y_train, df_full):
    # Evaluation
    risk_scores = model.predict(x_test)
    times = np.quantile(df_full["duration"][df_full["event"] == 1.0], horizons).tolist()

    surv_prob = np.row_stack([
        fn(times)
        for fn in model.predict_survival_function(x_test)
    ])
    brs = brier_score(y_train, y_test, surv_prob, times)[1]

    metric_dict = {}
    cis = []
    aucs = []
    for i, _ in enumerate(times):
        cis.append(
            concordance_index_ipcw(y_train, y_test, estimate=risk_scores, tau=times[i])[0]
        )
        aucs.append(cumulative_dynamic_auc(y_train, y_test, risk_scores, times[i])[0].item())
        metric_dict[f'{horizons[i]}_Ctd_ipcw'] = cis[i]
        metric_dict[f'{horizons[i]}_brier'] = brs[i]
        metric_dict[f'{horizons[i]}_auroc'] = aucs[i]

    return metric_dict


def run_experiments(model_name):
    results = []
    for dataset in DATASETS:
        seed_everything(random_state)

        x_train, x_val, x_test, y_train, y_val, y_test, df_train_raw, df_val_raw, df_test_raw, df_full, cols_standardize, cols_categorical = load_data(
            dataset, random_state=random_state)

        y_train = format_labels(y_train)
        y_test = format_labels(y_test)

        model = MODEL_DICT[model_name]()
        model.fit(x_train, y_train)

        result_dict = defaultdict(list)
        for i in range(nb_bootstrap):
            x_test_res = pd.DataFrame(x_test).sample(len(x_test), replace=True)
            y_test_res = pd.DataFrame(y_test).iloc[x_test_res.index]
            metric_dict = evaluate(model, x_test_res.values, y_test_res.to_records(index=False), y_train, df_full)

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

        row_str = ""
        for horizon in horizons:
            keys = [k for k in confi_dict.keys() if k.startswith(str(horizon))]
            results_at_horizon = [confi_dict[k] for k in keys]
            msg = [f"[{horizon * 100}%]"]
            for k, res in zip(keys, results_at_horizon):
                metric = k.split('_')[1]
                avg, interval = res
                msg.append(f"{metric}: {avg:.4f} ({interval:.4f})")
            row_str += (" ".join(msg) + "\n")

        results.append(row_str)

    tb = pt.PrettyTable(title=model_name)
    tb.field_names = DATASETS
    tb.add_row(results)
    print(tb)


run_experiments(model_name='cph')
run_experiments(model_name='rsf')