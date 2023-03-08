import torch
from torch import nn
from pycox.models.loss import nll_logistic_hazard

def assert_correct_input_shape(h):
    if len(h.shape) != 3:
        raise ValueError(f"h is of shape {h.shape}. It is expected that h is of shape (batch size, sequence_length, 1), as this is most amenable to use in training neural nets with pytorch.")

def assert_correct_output_shape(q, batch_size):
    if q.shape != torch.Size([batch_size, 1]):
        raise ValueError(f"q is of shape {q.shape}. It is expected that q is of shape (batch_size, 1)")

def survival_rate(h):
    """
    Given the predicted conditional hazard rate, this function estimates
    the survival rate.

    *input*:
    * `h`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`, as this is most amenable to use in training neural nets with pytorch.

    _output_:
    * `s`:
        - type: `torch.tensor`
        - estimated survival rate at time t.
        - note: `s.shape == (batch_size, 1)`
    """
    assert_correct_input_shape(h)
    s = (1-h).prod(dim=1)
    return s


def event_rate(h):
    """
    Given the predicted conditional hazard rate, this function estimates
    the event rate.

    *input*:
    * `h`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`, as this is most amenable to use in training neural nets with pytorch.

    _output_:
    * `w`:
        - type: `torch.tensor`
        - estimated survival rate at time t.
        - note: `w.shape == (batch_size, 1)`
    """
    assert_correct_input_shape(h)
    w = 1-survival_rate(h)
    return w


def event_time(h):
    """
    Given the predicted conditional hazard rate, this function estimates
    the probability that the event occurs at time t.

    *input*:
    * `h`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`, as this is most amenable to use in training neural nets with pytorch.

    _output_:
    * `p`:
        - type: `torch.tensor`
        - estimated probability of event at time t.
        - note: `p.shape == (batch_size, 1)`
    """
    assert_correct_input_shape(h)
    p = h[:, -1, :] * survival_rate(h[:, :-1, :])
    return p


def log_survival_rate(h):
    """
    Given the predicted conditional hazard rate, this function estimates
    the log survival rate.

    *input*:
    * `h`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`, as this is most amenable to use in training neural nets with pytorch.

    _output_:
    * `s`:
        - type: `torch.tensor`
        - estimated log survival rate at time t.
        - note: `s.shape == (batch_size, 1)`
    """
    assert_correct_input_shape(h)
    s = (1-h).log().sum(dim=1)
    return s


def log_event_rate(h):
    """
    Given the predicted conditional hazard rate, this function estimates
    the log event rate.

    *input*:
    * `h`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`, as this is most amenable to use in training neural nets with pytorch.

    _output_:
    * `w`:
        - type: `torch.tensor`
        - estimated log survival rate at time t.
        - note: `w.shape == (batch_size, 1)`
    """
    assert_correct_input_shape(h)
#     w = event_rate(h).log()                   # numerically unstable, darn probabilities
    w = (1 - log_survival_rate(h).exp()).log()  # numerically stable
    return w


def log_event_time(h):
    """
    Given the predicted conditional hazard rate, this function estimates
    the log probability that the event occurs at time t.

    *input*:
    * `h`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`, as this is most amenable to use in training neural nets with pytorch.

    _output_:
    * `p`:
        - type: `torch.tensor`
        - estimated log probability of event at time t.
        - note: `p.shape == (batch_size, 1)`
    """
    assert_correct_input_shape(h)
    p = torch.log(h[:, -1, :]) + log_survival_rate(h[:, :-1, :])
    return p


def event_time_loss(input, target=None):
    """
    Loss function applied to uncensored data in order
    to optimize the PDF of the true event time, z

    input:
    * `input`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`
    * `target`:
        - unused, only present to mimic pytorch loss functions

    output:
    * `evt_loss`:
        - type: `torch.tensor`
        - Loss associated with how wrong each predicted probability was at each time step
    """
    assert_correct_input_shape(input)
    evt_loss = -log_event_time(input).mean(dim=0).squeeze()
    return evt_loss


def event_rate_loss(input, target=None):
    """
    Loss function applied to uncensored data in order
    to optimize the CDF of the true event time, z

    input:
    * `input`:
        - type: `torch.tensor`,
        - predicted conditional hazard rate, at each observed time step.
        - note: `h.shape == (batch size, 1, 1)`
    * `target`:
        - unused, only present to mimic pytorch loss functions

    output:
    * `evr_loss`:
        - type: `torch.tensor`
        - Loss associated with how cumulative predicted probabilities differ from the ground truth labels.
    """
    assert_correct_input_shape(input)
    evr_loss = -log_event_rate(input).mean(dim=0).squeeze()
    return evr_loss


class LossTDSurv(nn.Module):
    def __init__(self, alpha=0.3, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, target=None):
        idx_durations, events = target.T

        L_z = event_time_loss(preds)

        c = 1.0 - events
        log_St = log_survival_rate(preds).squeeze()
        log_Wt = log_event_rate(preds).squeeze()
        L_c = -(c * log_St + (1 - c) * log_Wt).mean(dim=0).squeeze()

        def inverse_sigmoid(y):
            return torch.log(y / (1 - y))

        nll_loss = nll_logistic_hazard(inverse_sigmoid(preds[..., 0]), idx_durations.long(), events.float())

        # weighted average of L_z and L_c
        loss = (self.alpha * L_z) + ((1 - self.alpha) * L_c)
        loss += self.beta * nll_loss
        return loss
