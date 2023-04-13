import torch
from pycox.models.utils import pad_col
from torch import nn
from pycox.models.loss import nll_logistic_hazard, nll_pc_hazard_loss


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
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def event_time_loss(self, preds, idx_durations):
        """
        Event time loss (Note: compute only for uncensored cases)
        """
        h = preds[..., 0] # batch_size, time_step

        h_l = h.gather(1, idx_durations.unsqueeze(1)).view(-1) # batch_size
        h_padded = pad_col(h, where='start') # batch_size, time_step+1
        conditional_log_haz = torch.zeros_like(h)
        for i, idx in enumerate(idx_durations):
            conditional_log_haz[i, :idx] = torch.log(1 - h_padded[i, :idx])

        L_z = -(torch.log(h_l) + conditional_log_haz.sum(1))
        return torch.mean(L_z)

    def log_event_rate(self, preds, idx_durations):
        h = preds[..., 0]  # batch_size, time_step

        h_padded = pad_col(h, where='start')  # batch_size, time_step+1
        conditional_surv_log_prob = torch.zeros_like(h)
        for i, idx in enumerate(idx_durations):
            conditional_surv_log_prob[i, :idx] = torch.log(1 - h_padded[i, :idx])

        s = conditional_surv_log_prob.sum(1)
        event_rates = torch.clamp(1-s.exp(), min=1e-8)
        log_Wt = torch.log(event_rates)
        return log_Wt

    def log_survival_rate(self, preds, idx_durations):
        h = preds[..., 0]  # batch_size, time_step

        h_padded = pad_col(h, where='start')  # batch_size, time_step+1
        conditional_surv_log_prob = torch.zeros_like(h)
        for i, idx in enumerate(idx_durations):
            conditional_surv_log_prob[i, :idx] = torch.log(1 - h_padded[i, :idx])
        log_St = conditional_surv_log_prob.sum(1)
        return log_St

    def forward(self, preds, target=None):

        idx_durations, events, interval_frac = target.T
        idx_durations = idx_durations.long()

        L_z = self.event_time_loss(preds[events==1.0], idx_durations[events==1.0])

        c = 1.0 - events
        log_St = self.log_survival_rate(preds, idx_durations).squeeze()
        log_Wt = self.log_event_rate(preds, idx_durations).squeeze()
        L_c = -(c * log_St + (1 - c) * log_Wt).mean(dim=0).squeeze()

        def inverse_sigmoid(y):
            one_minus_y = (1-y) + 1e-12
            y_ = y + 1e-12
            return torch.log(y_ / one_minus_y)

        nll_loss = nll_logistic_hazard(inverse_sigmoid(preds[..., 0]), idx_durations.long(), events.float())

        # weighted average of L_z and L_c
        loss = (self.alpha * L_z) + ((1 - self.alpha) * L_c)
        loss += self.beta * nll_loss

        self.L_z = L_z
        self.L_c = L_c
        self.nll_loss = nll_loss

        return loss
