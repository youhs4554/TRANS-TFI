import os.path

import numpy as np
import pycox.models
import torch
from torch.utils.data import DataLoader
import torchtuples as tt
from matplotlib import pyplot as plt
from pycox.models.cox_time import MLPVanillaCoxTime
from baselines import models

from baselines.config import Config, BASELINE_MODEL_FAMILY
from baselines.datasets import get_TDSA_dataloader, get_test_TDSA_data, load_data
from baselines.loss import LossTDSurv
from baselines.utils import seed_everything

get_target = lambda df: (df['duration'].values, df['event'].values)


class PyCoxTrainer:
    def __init__(self, cfg: Config):
        self.dataset = None  # name of dataset
        self.labtrans = None
        self.taus = None

        self.train = None
        self.val = None
        self.test = None

        self.df_train_raw = None
        self.df_val_raw = None
        self.df_test_raw = None

        self.trained_model = None
        self.logger = None

        self.model_name = cfg.model_name
        self.net_class = tt.practical.MLPVanilla
        if cfg.model_name == 'CoxTime':
            self.net_class = MLPVanillaCoxTime
        self.model_class = getattr(pycox.models, cfg.model_name, None)
        self.time_range = cfg.time_range
        self.seq_len = cfg.seq_len
        self.interpolate_discrete_times = cfg.interpolate_discrete_times
        self.model_save_dir = cfg.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.cfg = cfg

        # for deterministic results
        seed_everything(cfg.random_state)

    def preprocess(self, dataset: str):
        # Data loading (train/valid)
        x_train, x_val, x_test, y_train, y_val, y_test, \
            df_train_raw, df_val_raw, df_test_raw, df_full, cols_standardize, cols_leave = \
            load_data(dataset)

        # evaluate the performance at the 25th, 50th and 75th event time quantile
        taus = np.quantile(df_full["duration"][df_full["event"] == 1.0], self.cfg.horizons).tolist()

        # Target label preprocessing (duration -> duration_idxs)
        labtrans = None
        if hasattr(self.model_class, 'label_transform'):
            init_labtrans = self.model_class.label_transform
            if self.time_range == 'truncated':
                # For truncated time ranges
                labtrans = init_labtrans(cuts=np.array([df_full["duration"].min()] + taus + [df_full["duration"].max()]))
            else:
                labtrans = init_labtrans(self.seq_len)

            y_train = labtrans.fit_transform(*get_target(df_train_raw))
            y_val = labtrans.transform(*get_target(df_val_raw))
        else:
            y_train = get_target(df_train_raw)
            y_val = get_target(df_val_raw)

        self.dataset = dataset
        self.labtrans = labtrans
        if dataset == 'dialysis':
            taus = [365 * 1, 365 * 3, 365 * 5] # evaluate at 1yr, 3yr, 5yr
            durations = np.sort(df_full["duration"][df_full["event"] == 1.0].values)
            horizons = [round(np.searchsorted(durations, t) / len(durations), 4) for t in taus]
            self.cfg.horizons = horizons
        self.taus = taus
        self.train = (x_train, y_train)
        self.val = (x_val, y_val)
        self.test = (x_test, y_test)

        self.df_train_raw = df_train_raw
        self.df_val_raw = df_val_raw
        self.df_test_raw = df_test_raw

    def make_net(self):
        n_features = self.train[0].shape[1]
        out_features = 1  # continuous time models have single output node (don't use labtrans)
        if self.labtrans is not None:
            out_features = self.labtrans.out_features  # discrete time models have multiple output nodes

        self.cfg.net_kwargs['in_features'] = n_features
        self.cfg.net_kwargs['out_features'] = out_features

        # Init networks
        net = self.net_class(**self.cfg.net_kwargs)
        net.to(self.cfg.device)
        return net

    def fit_and_predict(self, dataset, weight_file=''):
        self.preprocess(dataset)
        net = self.make_net()
        kwargs = {}
        if self.cfg.model_name == "DeepHitSingle":
            kwargs.update({"alpha": self.cfg.alpha, "sigma": self.cfg.sigma})
        if self.labtrans is not None:
            kwargs["duration_index"] = self.labtrans.cuts

        fitting = True
        if weight_file:
            assert os.path.exists(weight_file)
            fitting = False
            state_dict = torch.load(weight_file)
            net.load_state_dict(state_dict)

        model = self.model_class(net, tt.optim.Adam(lr=self.cfg.lr, weight_decay=self.cfg.weight_decay), **kwargs)

        if hasattr(model, 'compute_baseline_hazards'):
            model.training_data = self.train

        if fitting:
            self.logger.info(
                f"[{self.cfg.model_name}@{dataset}] time_range={self.cfg.time_range}, L={self.cfg.seq_len}, interpolate={self.interpolate_discrete_times}")
            verbose = False if self.cfg.silent_fit else True
            es_patience = self.cfg.es_patience

            if isinstance(self.train, DataLoader) and isinstance(self.val, DataLoader):
                log = model.fit_dataloader(self.train, epochs=self.cfg.n_ep, callbacks=[tt.callbacks.EarlyStopping(patience=es_patience)],
                                           val_dataloader=self.val, verbose=verbose)
            else:
                if dataset == 'dialysis':
                    print()
                log = model.fit(*self.train, epochs=self.cfg.n_ep, callbacks=[tt.callbacks.EarlyStopping(patience=es_patience)],
                                val_data=self.val, verbose=verbose)
            if self.cfg.show_plot:
                log.plot();
                plt.show()
            history = log.to_pandas()

        # Inference
        x_test, _ = self.test
        if self.interpolate_discrete_times:
            surv = model.interpolate(100).predict_surv(x_test)
        else:
            if hasattr(model, 'compute_baseline_hazards'):
                _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test)

        self.trained_model = model
        model_save_path = self.model_save_dir / f"{self.model_name}_{self.time_range}_L{self.seq_len}_{dataset}.pth"
        self.trained_model.save_model_weights(model_save_path)

        return surv, model, history

class PyTorchTrainer(PyCoxTrainer):
    def __init__(self, cfg: Config):
        super(PyTorchTrainer, self).__init__(cfg)
        self.net_class = models.TDSA
        beta = 1.0
        if not cfg.model_name == 'BTDSA':
            beta = 0.0

        self.model_class = lambda net, optimizer, duration_index: models.PyCoxWrapper(net, LossTDSurv(beta=beta),
                                                                                      optimizer,
                                                                                      duration_index=duration_index)

    def preprocess(self, dataset: str):
        # Data loading (train/valid)
        train_loader, val_loader = get_TDSA_dataloader(dataset=dataset, seq_len=self.cfg.seq_len, random_state=self.cfg.random_state)

        x_test, durations_test, events_test, test_ds = get_test_TDSA_data(dataset=dataset,
                                                                          seq_len=self.cfg.seq_len, random_state=self.cfg.random_state)  # get test data

        labtrans = getattr(train_loader.dataset, 'labtrans', None)
        taus = train_loader.dataset.taus

        self.dataset = dataset
        self.labtrans = labtrans
        self.taus = taus

        self.train = train_loader
        self.val = val_loader
        self.test = (x_test, np.c_[durations_test, events_test])

        self.df_train_raw = train_loader.dataset.df_raw
        self.df_val_raw = val_loader.dataset.df_raw
        self.df_test_raw = test_ds.df_raw

    def make_net(self):
        train_ds = self.train.dataset
        embeddings = models.get_embeddings(train_ds.n_embeddings)

        self.cfg.net_kwargs['n_features'] = train_ds.n_features + 1  # +1 for time features
        self.cfg.net_kwargs['output_size'] = 1  # each time step has single node (=sigmoid)
        self.cfg.net_kwargs['n_durations'] = self.cfg.seq_len if self.cfg.time_range == 'full' else self.cfg.seq_len + 2
        self.cfg.net_kwargs['embeddings'] = embeddings
        net = self.net_class(**self.cfg.net_kwargs)
        net.to(self.cfg.device)
        return net


def init_trainer(cfg):
    if cfg.model_name in BASELINE_MODEL_FAMILY:
        trainer_class = PyCoxTrainer
    elif cfg.model_name == 'DRSA':
        trainer_class = PyTorchTrainer
    else:
        raise NotImplementedError
    return trainer_class(cfg)