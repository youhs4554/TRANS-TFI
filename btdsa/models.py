import pandas as pd
import torch
import torch.nn as nn
from typing import List

from pycox.models.interpolation import InterpolateLogisticHazard
from pycox.models.utils import pad_col
import torchtuples as tt

def get_embeddings(n_embeddings: List, embedding_size: int = 50, device=None):
    """
    Get embeddings (look-up matrix) for categorical features
    :param n_embeddings: list composed with rows of embedding matrix
    :param embedding_size: size of each embedding vector
    :param device: device to use (cpu / gpu / mps)
    :return:
    """
    layers = []
    for nb_emb in n_embeddings:
        emb = torch.nn.Embedding(nb_emb, embedding_size, device=device)
        emb.weight.data.uniform_(-0.1, 0.1)
        layers.append(emb)
    return layers

class TDSA(nn.Module):
    """
    Time Dependent Survival Analysis model.
    A relatively shallow net, characterized by an Bidirectional LSTM layer followed by a single Linear layer.
    """

    def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            n_layers: int,
            embeddings: List[nn.Embedding],
            output_size: int = 1,
            bidirectional: bool = True
    ):
        """
        inputs:
        * `n_features`
            - size of the input to the LSTM (number of features)
        * `hidden_dim`:
            - size (dimension) of the hidden state in LSTM
        * `n_layers`:
            - number of layers in LSTM
        * `embeddings`:
            - list of nn.Embeddings for each categorical variable
            - It is assumed the 1st categorical feature corresponds with the 0th feature,
              the 2nd corresponds with the 1st feature, and so on.
        * `output_size`:
            - size of the linear layer's output, which should always be 1, unless altering this model
        * `bidirectional`:
            - use bidirectional or not
        """
        super(TDSA, self).__init__()

        # hyper params
        self.n_features = n_features
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embeddings
        self.embeddings = embeddings

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            sum([emb.embedding_dim for emb in self.embeddings])
            + self.n_features
            - len(self.embeddings),
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        if bidirectional:
            hidden_dim = hidden_dim * 2
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

        # making sure embeddings get trained
        self.params_to_train = nn.ModuleList(self.embeddings)

    def forward(self, X: torch.tensor):
        """
        input:
        * `X`
            - input features of shape (batch_size, sequence length, self.n_features)

        output:
        * `out`:
            - the DRSA model's predictions at each time step, for each observation in batch
            - out is of shape (batch_size, sequence_length, 1)
        """
        # concatenating embedding and numeric features
        all_embeddings = [
            emb(X[:, :, i].long()) for i, emb in enumerate(self.embeddings)
        ]
        other_features = X[:, :, len(self.embeddings):]
        all_features = torch.cat(all_embeddings + [other_features.float()], dim=-1)

        # passing input and hidden into model (hidden initialized as zeros)
        out, hidden = self.lstm(all_features.float())

        # passing to linear layer to reshape for predictions
        out = self.sigmoid(self.fc(out))

        return out

class PyCoxWrapper(tt.Model):
    """Wrapper class for pycox API compatibility
    """
    _steps = 'post'

    def __init__(self, net, loss=None, optimizer=None, device=None, duration_index=None):
        self.duration_index = duration_index
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self):
        """
        Array of durations that defines the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.

        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0):
        # pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        preds = pad_col(preds[..., 0])[:, :-1]
        # surv = (1-preds).cumprod(1)
        # surv = 1 - preds.cumsum(1)
        surv = (1 - preds).add(1e-7).log().cumsum(1).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                    num_workers=0):
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds[..., 0])[:, :-1]
        return tt.utils.array_or_tensor(pmf, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        """Use interpolation for predictions.
        There is only one scheme:
            `const_pdf` and `lin_surv` which assumes pice-wise constant PMF in each interval (linear survival).

        Keyword Arguments:
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
            scheme {str} -- Type of interpolation {'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})

        Returns:
            [InterpolateLogisticHazard] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateLogisticHazard(self, scheme, duration_index, sub)
