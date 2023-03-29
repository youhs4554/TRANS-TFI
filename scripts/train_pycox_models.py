import warnings
from itertools import product

warnings.filterwarnings('ignore')

import os, sys
sys.path.append(os.path.abspath('../btdsa'))
from btdsa.config import Config, BASELINE_MODEL_FAMILY
from btdsa.eval_utils import EvalSurv
from btdsa.train_utils import init_trainer
from btdsa.utils import create_logger

logger = create_logger(logs_dir='../experiments/logs')



def run_experiment(model_name, time_range='full'):
    ev = EvalSurv()  # custom evaluation interface

    cfg = Config
    cfg.model_name = model_name
    cfg.time_range = time_range
    cfg.setup()

    for dataset in cfg.list_of_datasets:
        trainer = init_trainer(cfg)
        trainer.logger = logger  # assign logger

        surv, model = trainer.fit_and_predict(dataset)
        ev.trainer = trainer
        x_test, y_test = trainer.test
        ev.evaluate(surv, x_test, y_test)
    ev.report()  # log and report results in beautiful tables

# Comparison with Baselines; CoxPH(=DeepSurv), DeepHitSingle
for model_name, time_range in product(BASELINE_MODEL_FAMILY, ["full", "truncated"]):
    if model_name == 'PCHazard' and time_range == 'truncated':
        continue
    run_experiment(model_name, time_range)