import warnings
from itertools import product

warnings.filterwarnings('ignore')

import os, sys

sys.path.append(os.path.abspath('../baselines'))
from baselines.config import Config, BASELINE_MODEL_FAMILY, TIME_INJECTION_MODEL_FAMILY
from baselines.eval_utils import Evaluator
from baselines.train_utils import init_trainer
import mlflow

def run_experiment(model_name, time_range='full'):
    ev = Evaluator()  # custom evaluation interface

    cfg = Config
    cfg.model_name = model_name
    cfg.time_range = time_range
    cfg.setup()

    for dataset in cfg.list_of_datasets:
        experiment = mlflow.get_experiment_by_name(dataset)
        if experiment is None:
            experiment_id = mlflow.create_experiment(dataset)
            experiment = mlflow.get_experiment(experiment_id)

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{cfg.model_name}_{cfg.time_range}_L={cfg.seq_len}"):
            mlflow.log_param('time_range', cfg.time_range)
            mlflow.log_param('L', cfg.seq_len)
            trainer = init_trainer(cfg)

            trainer.fit_and_predict(dataset)
            ev.trainer = trainer
            x_test, y_test = trainer.test
            ev.evaluate(x_test, y_test)
    ev.report()  # log and report results in beautiful tables


# Comparison with Baselines; CoxPH(=DeepSurv), DeepHitSingle
for model_name, time_range in product(BASELINE_MODEL_FAMILY + TIME_INJECTION_MODEL_FAMILY, ["full", "truncated"]):
    if model_name == 'PCHazard' and time_range == 'truncated':
        continue
    run_experiment(model_name, time_range)
