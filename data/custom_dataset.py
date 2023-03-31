from pathlib import Path

import pandas as pd
import json

from sklearn.impute import SimpleImputer


class _CustomDataset:
    _dataset_dir = Path(__file__).parent
    # datasets can be extended if needed
    _datasets = {
        'dialysis': 'prepared_full.xlsx'
    }
    _meta = {
        'dialysis': 'column_types.json'
    }

    def __init__(self):
        meta_path = self._dataset_dir / self.name / self._meta[self.name]
        column_types = json.load(open(meta_path))
        self.cols_numerical = column_types['num']
        self.cols_categorical = column_types['cat']

    def read_df(self):
        raise NotImplementedError


class _Dialysis(_CustomDataset):
    name = 'dialysis'

    def read_df(self):
        dataset_path = self._dataset_dir / self.name / self._datasets[self.name]
        df = pd.read_excel(dataset_path)
        df = df[self.cols_numerical + self.cols_categorical + ['survival_days', 'death']]

        df[self.cols_numerical] = SimpleImputer(strategy='mean').fit_transform(df[self.cols_numerical])
        df[self.cols_categorical] = SimpleImputer(strategy='most_frequent').fit_transform(df[self.cols_categorical])
        df.rename({'death': 'event', 'survival_days': 'duration'}, axis=1, inplace=True)
        return df


dialysis = _Dialysis()