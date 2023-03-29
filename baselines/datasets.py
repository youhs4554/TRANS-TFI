import pandas as pd
from sklearn.preprocessing import StandardScaler
import pycox.datasets as dt


def load_data(dataset, random_state=1234):
    if dataset == 'flchain':
        cols_standardize = ['age', 'creatinine', 'kappa', 'lambda']
        cols_leave = ['sex', 'sample.yr', 'mgus', 'flc.grp']
    elif dataset == 'metabric':
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
    elif dataset == 'gbsg':
        cols_standardize = ['x3', 'x4', 'x5', 'x6']
        cols_leave = ['x0', 'x1', 'x2']
    else:
        raise NotImplementedError

    cols_tgt = ['duration', 'event']

    # Split train/val/test
    df_full = getattr(dt, dataset).read_df()
    if dataset == 'flchain':
        df_full.rename({'futime': 'duration', 'death': 'event'}, axis=1, inplace=True)
        df_full = df_full.astype('float32')
    df_feats = df_full.drop(cols_tgt, axis=1)
    df_labels = df_full[cols_tgt]

    df_feats_standardize = df_feats[cols_standardize]
    df_feats_standardize_disc = StandardScaler().fit_transform(df_feats_standardize)
    df_feats_standardize_disc = pd.DataFrame(df_feats_standardize_disc, columns=cols_standardize)

    df_feats = pd.concat([df_feats[cols_leave], df_feats_standardize_disc], axis=1)
    df_full = pd.concat([df_feats, df_labels], axis=1)
    max_duration_idx = df_full["duration"].argmax()
    df_test = df_full.drop(max_duration_idx).sample(frac=0.3, random_state=random_state)
    df_train = df_full.drop(df_test.index)
    df_val = df_train.drop(max_duration_idx).sample(frac=0.1, random_state=random_state)
    df_train = df_train.drop(df_val.index)

    # Target info
    y_train = df_train[cols_tgt].values
    y_val = df_val[cols_tgt].values
    y_test = df_test[cols_tgt].values

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

    return x_train, x_val, x_test, y_train, y_val, y_test, df_train_raw, df_val_raw, df_test_raw, df_full, cols_standardize, cols_leave
