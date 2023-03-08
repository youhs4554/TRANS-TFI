import logging
import os
import time

import numpy as np
import scipy.stats as st
import torch
from pycox.evaluation import EvalSurv


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
def bootstrap_eval(model, x_test, durations_test, events_test, interpolate_discrete_times=True, nb_bootstrap=100):
    nb_samples = len(x_test)
    metrics = []

    def resample():
        indices = np.random.choice(range(nb_samples), nb_samples, replace=True)
        return x_test[indices], durations_test[indices], events_test[indices]

    for i in range(nb_bootstrap):
        x_test_res, durations_test_res, events_test_res = resample()
        if interpolate_discrete_times:
            surv = model.interpolate(100).predict_surv_df(x_test_res)
        else:
            surv = model.predict_surv_df(x_test_res)
        ev = EvalSurv(surv, durations_test_res, events_test_res, censor_surv='km')
        cindex = ev.concordance_td('antolini')
        metrics.append(cindex)

    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }
