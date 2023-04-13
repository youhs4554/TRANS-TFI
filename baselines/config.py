import os.path
from pathlib import Path

import torch
from easydict import EasyDict

BASELINE_MODEL_FAMILY = [
    # Discrete-Time Models,
    "DeepHitSingle", "LogisticHazard", "PMF", "MTLR", "BCESurv",
    # Continuous-Time Models
    "CoxPH", "PCHazard",
]

TIME_INJECTION_MODEL_FAMILY = [
    # Deep Recurrent Survival Analysis (https://arxiv.org/abs/1809.02403)
    "DRSA"
]


class Config(EasyDict):
    model_name = 'DeepHitSingle'  # which model to use
    time_range = 'full'  # valid only for discrete-time models (options: full | truncated)
    random_state = 1234  # static random state for deterministic results
    n_ep = 3000  # number of epochs
    model_save_dir = Path('./model_dir')
    model_save_dir.mkdir(exist_ok=True)

    @classmethod
    def setup(cls):
        cls.horizons = [.25, .5, .75]  # truncated time horizons 25%, 50%, 75%
        cls.list_of_datasets = ['gbsg', 'metabric', 'dialysis']  # name of dataset

        assert cls.model_name in BASELINE_MODEL_FAMILY + TIME_INJECTION_MODEL_FAMILY

        cls.lr = 1e-3  # learning rate
        cls.weight_decay = 0.0  # weight decay strength
        cls.seq_len = 20  # default length
        if cls.time_range == 'truncated':
            cls.seq_len = 3
        cls.show_plot = False  # if true, plot learning curves
        cls.silent_fit = True  # if true, do not show training progress
        cls.nb_bootstrap = 10  # number of bootstrap evaluations
        cls.es_patience = 5

        if cls.model_name == "DeepHitSingle":
            # Used for DeepHit loss
            cls.alpha = 0.2  # weights for combining nll and ranking loss
            cls.sigma = 0.1  # used in ranking loss

        # if true, interpolate discrete-time model outputs for evaluation
        cls.interpolate_discrete_times = (cls.model_name in ['DeepHitSingle', 'LogisticHazard', 'PMF', 'MTLR',
                                                             'BCESurv', 'DRSA'])

        if not torch.backends.mps.is_available():
            cls.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            cls.device = torch.device("mps") # Apple's M1 mac setup

        cls.logs_dir = Path('./logs')
        cls.logs_dir.mkdir(exist_ok=True)

        if cls.model_name in BASELINE_MODEL_FAMILY:
            cls.net_kwargs = dict(
                num_nodes=[32, 32],
                batch_norm=True,
                dropout=0.1,
                output_bias=False
            )
        elif cls.model_name in TIME_INJECTION_MODEL_FAMILY:
            cls.net_kwargs = dict(
                hidden_dim=64,
                n_layers=3,  # number of LSTM layer(s)
            )
        else:
            raise NotImplementedError(f"Unsupported model_name={cls.model_name}")
