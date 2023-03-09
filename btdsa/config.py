from pathlib import Path

import torch
from easydict import EasyDict

BASELINE_MODEL_FAMILY = [
    # Discrete-Time Models,
    "DeepHitSingle", "LogisticHazard", "LogisticHazard", "PMF", "MTLR", "BCESurv",
    # Continuous-Time Models
    "CoxPH", "PCHazard", "CoxTime", "CoxCC"
]

TDSA_MODEL_LIST = [
    # Deep Recurrent Survival Analysis model (paper: https://arxiv.org/abs/1809.02403)
    "DRSA",
    # Our proposed model (BTDSA: Bidirectional Time Dependent Survival Analysis)
    "BTDSA"
]


class Config(EasyDict):
    model_name = 'BTDSA'  # which model to use
    time_range = 'full'  # valid only for discrete-time models (options: full | truncated)
    random_state = 1234  # static random state for deterministic results
    n_ep = 3000  # number of epochs

    @classmethod
    def setup(cls):
        cls.horizons = [.25, .5, .75]  # truncated time horizons 25%, 50%, 75%
        cls.list_of_datasets = ['gbsg', 'metabric', 'support']  # name of dataset

        assert cls.model_name in BASELINE_MODEL_FAMILY + TDSA_MODEL_LIST

        cls.lr = 1e-3  # learning rate
        cls.weight_decay = 1e-4  # weight decay strength
        cls.seq_len = 20  # default length
        if cls.time_range == 'truncated':
            cls.seq_len = 3
        cls.show_plot = False  # if true, plot learning curves
        cls.silent_fit = True  # if true, do not show training progress
        cls.nb_bootstrap = 100  # number of bootstrap evaluations

        if cls.model_name == "DeepHitSingle":
            # Used for DeepHit loss
            cls.alpha = 0.2  # weights for combining nll and ranking loss
            cls.sigma = 0.1  # used in ranking loss

        # if true, interpolate discrete-time model outputs for evaluation
        cls.interpolate_discrete_times = (cls.model_name in ['DeepHitSingle', 'LogisticHazard', 'PMF', 'MTLR', 'BCESurv']) and (cls.time_range == 'full')

        if not torch.backends.mps.is_available():
            cls.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Apple's M1 mac setup
        else:
            cls.device = torch.device("mps")

        cls.logs_dir = Path('./logs') / cls.model_name

        if cls.model_name in BASELINE_MODEL_FAMILY:
            cls.net_kwargs = dict(
                num_nodes=[32, 32],
                batch_norm=True,
                dropout=0.1,
                output_bias=False
            )
        elif cls.model_name in ['BTDSA', 'DRSA']:
            use_BTDSA = (cls.model_name == 'BTDSA')
            cls.net_kwargs = dict(
                hidden_dim=64,
                n_layers=3,  # number of LSTM layer(s)
                LSTM_dropout=0.0,  # LSTM dropout
                bidirectional=use_BTDSA
            )
        else:
            raise NotImplementedError(f"Unsupported model_name={cls.model_name}")
