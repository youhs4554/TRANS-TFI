import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pycox.datasets as dt
from pycox.preprocessing import label_transforms
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(dataset, random_state=1234):
    if dataset == 'support':
        cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        cols_categorical = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    elif dataset == 'metabric':
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_categorical = ['x4', 'x5', 'x6', 'x7']
    elif dataset == 'gbsg':
        cols_standardize = ['x3', 'x4', 'x5', 'x6']
        cols_categorical = ['x0', 'x1', 'x2']
    else:
        raise NotImplementedError

    cols_tgt = ['duration', 'event']

    # Split train/val/test
    df_full = getattr(dt, dataset).read_df()
    df_feats = df_full.drop(cols_tgt, axis=1)
    df_labels = df_full[cols_tgt]

    df_feats_standardize = df_feats[cols_standardize]
    df_feats_standardize_disc = StandardScaler().fit_transform(df_feats_standardize)
    df_feats_standardize_disc = pd.DataFrame(df_feats_standardize_disc, columns=cols_standardize)

    for col in cols_categorical:
        df_feats[col] = LabelEncoder().fit_transform(df_feats[col]).astype('float32')
    df_feats = pd.concat([df_feats[cols_categorical], df_feats_standardize_disc], axis=1)
    df_full = pd.concat([df_feats, df_labels], axis=1)
    max_duration_idx = df_full["duration"].argmax()
    df_test = df_full.drop(max_duration_idx).sample(frac=0.3, random_state=random_state)
    df_train = df_full.drop(df_test.index)
    df_val = df_train.drop(max_duration_idx).sample(frac=0.1, random_state=random_state)
    df_train = df_train.drop(df_val.index)

    # Target info
    y_train = df_train[cols_tgt].values.astype('float32')
    y_val = df_val[cols_tgt].values.astype('float32')
    y_test = df_test[cols_tgt].values.astype('float32')

    df_train_raw = df_train
    df_val_raw = df_val
    df_test_raw = df_test

    # Input Covariates
    df_train = df_train.drop(cols_tgt, axis=1)
    df_val = df_val.drop(cols_tgt, axis=1)
    df_test = df_test.drop(cols_tgt, axis=1)

    x_train = df_train.values
    x_val = df_val.values
    x_test = df_test.values

    return x_train, x_val, x_test, y_train, y_val, y_test, df_train_raw, df_val_raw, df_test_raw, df_full, cols_standardize, cols_categorical

class TDSADataset(Dataset):
    def __init__(self, dataset, seq_len=20, phase='train', random_state=1234):
        super().__init__()

        x_train, x_val, x_test, y_train, y_val, y_test, df_train_raw, df_val_raw, df_test_raw, df_full, cols_standardize, cols_leave = load_data(dataset, random_state=random_state)

        self.cols_standardize = cols_standardize
        self.cols_leave = cols_leave
        self.seq_len = seq_len

        target_parse_fn = TDSADataset.get_target

        # For truncated time ranges
        horizons = [.25, .5, .75]  # truncated time horizons 25%, 50%, 75%
        # evaluate the performance at the 25th, 50th and 75th event time quantile
        taus = list(np.quantile(df_full["duration"][df_full["event"] == 1.0], horizons))
        setattr(self, 'taus', taus)

        if seq_len == 3:
            labtrans = label_transforms.LabTransDiscreteTime(cuts=np.array([df_full["duration"].min()] + taus + [df_full["duration"].max()]))
            setattr(self, 'seq_len', len(labtrans.cuts))
        else:
            labtrans = label_transforms.LabTransDiscreteTime(seq_len)
            setattr(self, 'seq_len', seq_len)

        # Target label preprocessing (duration -> duration_idxs)
        y_train = np.c_[labtrans.fit_transform(*target_parse_fn(y_train))]
        y_val = np.c_[labtrans.transform(*target_parse_fn(y_val))]

        setattr(self, 'labtrans', labtrans)

        if phase == 'train':
            self.X, self.Y, self.df_raw = x_train, y_train, df_train_raw
        if phase == 'val':
            self.X, self.Y, self.df_raw = x_val, y_val, df_val_raw
        if phase == 'test':
            self.X, self.Y, self.df_raw = x_test, y_test, df_test_raw


    @property
    def numeric_columns(self):
        return self.cols_standardize

    @property
    def categorical_columns(self):
        return self.cols_leave

    @property
    def n_features(self):
        return self.X.shape[1]

    @property
    def n_embeddings(self):
        return [len(np.unique(self.X[:, ix])) for ix in range(len(self.categorical_columns))]

    @staticmethod
    def get_target(y):
        durations, events = y.T
        return durations, events

    def __getitem__(self, ix):
        x = torch.from_numpy(self.X[ix])
        y = torch.from_numpy(self.Y[ix])

        return x, y

    def __len__(self):
        return len(self.X)


def get_TDSA_dataloader(dataset: str, seq_len: int = 20, batch_size: int = 256, random_state: int = 1234):
    """
    :param dataset: name of dataset
    :param seq_len: maximum sequence length
    :param batch_size: size of each batch
    :return: tuple of (train_loader, valid_loader)
    """

    train_ds = TDSADataset(dataset, seq_len=seq_len, phase='train', random_state=random_state)
    val_ds = TDSADataset(dataset, seq_len=seq_len, phase='val', random_state=random_state)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_TDSA_data(dataset: str, seq_len: int = 20, random_state: int = 1234):
    test_ds = TDSADataset(dataset=dataset, seq_len=seq_len, phase='test', random_state=random_state)

    # Evaluation
    x_test = []
    y_test = []

    for x, y in test_ds:
        x_test.append(x.numpy())
        if isinstance(y, tuple):
            y = torch.stack(y)
        y_test.append(y.numpy())

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    durations_test, events_test = y_test.T

    return x_test, durations_test, events_test, test_ds
