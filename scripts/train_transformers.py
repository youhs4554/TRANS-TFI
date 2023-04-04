#!/usr/bin/env python
# coding: utf-8

# Run SurvTRACE on GBSG, Metabric, Support  datasets

import warnings

import mlflow
import numpy as np

from SurvTRACE.survtrace.custom_loss import LossTDSurv
from SurvTRACE.survtrace.losses import NLLPCHazardLoss

warnings.filterwarnings('ignore')

import os, sys

sys.path.append(os.path.abspath('../SurvTRACE'))

import matplotlib.pyplot as plt

from SurvTRACE.survtrace.dataset import load_data
from SurvTRACE.survtrace.evaluate_utils import Evaluator
from SurvTRACE.survtrace.utils import set_random_seed
from SurvTRACE.survtrace.model import SurvTraceSingle
from SurvTRACE.survtrace.train_utils import Trainer
from SurvTRACE.survtrace.config import STConfig
import prettytable as pt

DATASETS = ['gbsg', 'metabric', 'dialysis']

from baselines.utils import create_logger

logger = create_logger('./logs_survtrace')

headers = DATASETS

results = []


def run_experiment(dataset, custom_training=True, show_plot=False):
    assert dataset in DATASETS
    if custom_training:
        model_name = "TRANS-TFI"
    else:
        model_name = "SurvTrace"
    logger.info(f"Training {model_name}@{dataset}...")

    # define the setup parameters
    STConfig.data = dataset
    STConfig.early_stop_patience = 5
    STConfig.custom_training = custom_training

    metrics = [NLLPCHazardLoss(), ]
    if STConfig['custom_training']:
        metrics = [LossTDSurv(), ]

    seed = STConfig.seed
    set_random_seed(seed)

    hparams = {
        'batch_size': 64,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
        'epochs': 200,
    }
    experiment = mlflow.get_experiment_by_name(dataset)
    if experiment is None:
        experiment_id = mlflow.create_experiment(dataset)
        experiment = mlflow.get_experiment(experiment_id)

    with mlflow.start_run(experiment_id=experiment.experiment_id,
                          run_name=model_name):
        # load data
        df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)

        if dataset == 'dialysis':
            times = [365 * 1, 365 * 3, 365 * 5, 365 * 7] # evaluate at 1yr, 3yr, 5yr, 7yr
            horizons = [ f"{i}yr" for i in [1,3,5,7] ]
        else:
            times = STConfig['duration_index'][1:-1] # evaluate at 25%, 50%, 75% durations (default)
            horizons = STConfig['horizons']

        # get model
        model = SurvTraceSingle(STConfig)

        # initialize a trainer
        trainer = Trainer(model, metrics=metrics)
        train_loss, val_loss = trainer.fit((df_train, df_y_train), (df_val, df_y_val),
                                           batch_size=hparams['batch_size'],
                                           epochs=hparams['epochs'],
                                           learning_rate=hparams['learning_rate'],
                                           weight_decay=hparams['weight_decay'])

        # evaluate model
        evaluator = Evaluator(df, df_train.index)
        result_dict = evaluator.eval(model, (df_test, df_y_test), times=times, horizons=horizons, confidence=.95, nb_bootstrap=10)

        # Messages for pretty table summary
        cindex_avg, cindex_interval = result_dict.pop('C-td-full')
        row_str = f"C-td (full): {cindex_avg:.4f} ({cindex_interval:.4f})\n"
        mlflow.log_metric('Ctd__avg', cindex_avg)
        mlflow.log_metric('Ctd__interval', cindex_interval)

        for horizon in horizons:
            keys = [k for k in result_dict.keys() if k.startswith(str(horizon))]
            results_at_horizon = [result_dict[k] for k in keys]
            if dataset == 'dialysis':
                msg = [f"[{horizon}]"]
            else:
                msg = [f"[{round(horizon * 100, 4)}%]"]
            for k, res in zip(keys, results_at_horizon):
                metric = k[k.find('_')+1:]
                avg, interval = res
                msg.append(f"{metric}: {avg:.4f} ({interval:.4f})")
                mlflow.log_metric(str(horizon) + '_' + metric + '__avg', avg)
                mlflow.log_metric(str(horizon) + '_' + metric + '__interval', interval)
            row_str += (" ".join(msg) + "\n")
        results.append(row_str)

        if show_plot:
            # show training curves
            plt.plot(train_loss, label='train')
            plt.plot(val_loss, label='val')
            plt.legend(fontsize=20)
            plt.xlabel('epoch', fontsize=20)
            plt.ylabel('loss', fontsize=20)
            plt.show()


def fit_report(custom_training):
    global results

    for dataset in DATASETS:
        run_experiment(dataset, custom_training=custom_training, show_plot=False)

    if custom_training:
        title = "TRANS-TFI"
    else:
        title = "SurvTrace"

    tb = pt.PrettyTable(title=title)
    tb.field_names = headers
    tb.add_row(results)
    logger.info(tb)
    results = []


fit_report(custom_training=False)
fit_report(custom_training=True)

