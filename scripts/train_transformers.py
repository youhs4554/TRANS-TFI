#!/usr/bin/env python
# coding: utf-8

# Run SurvTRACE on GBSG, Metabric, Support  datasets

import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

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

headers = DATASETS

results = []


def run_experiment(dataset, custom_training=True, show_plot=False):
    assert dataset in DATASETS
    if custom_training:
        model_name = "TRANS-TFI"
    else:
        model_name = "SurvTrace"
    print(f"Training {model_name}@{dataset}...")

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

        if STConfig.custom_training:
            mlflow.log_param('injection_type', STConfig.injection_type)
        mlflow.log_param('L', STConfig.out_feature)
        mlflow.log_param('embedding_size', STConfig.hidden_size)

        # initialize a trainer
        trainer = Trainer(model, dataset, metrics=metrics)
        history = trainer.fit((df_train, df_y_train), (df_val, df_y_val),
                                           batch_size=hparams['batch_size'],
                                           epochs=hparams['epochs'],
                                           learning_rate=hparams['learning_rate'],
                                           weight_decay=hparams['weight_decay'])
        history = pd.DataFrame.from_dict(history)

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
            with plt.style.context('ggplot'):
                history.plot()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()

        history.to_csv(Path('./logs') / f"{model_name}_{dataset}.csv",
                       index_label='epoch')

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
    print(tb)
    results = []


fit_report(custom_training=False)
fit_report(custom_training=True)

