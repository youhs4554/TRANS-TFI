import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from pycox.evaluation import EvalSurv
from pycox import utils
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc


def seed_everything(random_state=1234):
    """
    Fix randomness for deterministic results
    :param random_state: random state generating random numbers
    :return: None
    """

    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)  # multi-GPU


def create_logger(logs_dir):
    ''' Performs creating logger
    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    os.makedirs(logs_dir, exist_ok=True)
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


@torch.no_grad()
def bootstrap_eval(model, x_test, durations_test, events_test, et_train, taus, horizons,
                   interpolate_discrete_times=True, nb_bootstrap=100):
    nb_samples = len(x_test)
    result_dict = defaultdict(list)  # C-td(IPCW)/Brier/AUC of 25%,50%,75% horizons

    def resample():
        x_test_res = pd.DataFrame(x_test).sample(nb_samples, replace=True)
        durations_test_res = pd.Series(durations_test).iloc[x_test_res.index]
        events_test_res = pd.Series(events_test).loc[x_test_res.index]
        return x_test_res.values, durations_test_res.values, events_test_res.values

    for i in range(nb_bootstrap):
        x_test_res, durations_test_res, events_test_res = resample()
        if interpolate_discrete_times:
            surv = model.interpolate(100).predict_surv_df(x_test_res)
        else:
            surv = model.predict_surv_df(x_test_res)
        ev = EvalSurv(surv, durations_test_res, events_test_res, censor_surv='km')
        cindex = ev.concordance_td('antolini')
        result_dict['C-td-full'].append(cindex)

        idx = utils.idx_at_times(surv.index, taus, 'post')
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